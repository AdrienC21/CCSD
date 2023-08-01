#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""sampler.py: code for sampling from the model.
"""

import abc
import os
import time
import math

import pickle
import torch
import wandb
from easydict import EasyDict
from moses.metrics.metrics import get_all_metrics

from src.utils.logger import (
    Logger,
    set_log,
    start_log,
    train_log,
    sample_log,
    check_log,
)
from src.utils.loader import (
    load_ckpt,
    load_data,
    load_seed,
    load_device,
    load_model_from_ckpt,
    load_ema_from_ckpt,
    load_sampling_fn,
    load_eval_settings,
)
from src.utils.graph_utils import adjs_to_graphs, quantize, quantize_mol
from src.utils.plot import (
    plot_graphs_list,
    save_graph_list,
    plot_cc_list,
    save_cc_list,
    plot_molecule_list,
    save_molecule_list,
    plot_3D_molecule,
    rotate_molecule_animation,
)
from src.evaluation.stats import eval_graph_list
from src.utils.mol_utils import (
    gen_mol,
    mols_to_smiles,
    load_smiles,
    canonicalize_smiles,
    mols_to_nx,
)
from src.utils.cc_utils import (
    cc_from_incidence,
    convert_CC_to_graphs,
    mols_to_cc,
    init_flags,
)
from src.utils.mol_utils import is_molecular_config


class Sampler(abc.ABC):
    """Abstract class for Sampler objects."""

    def __init__(self, config: EasyDict) -> None:
        """Initialize the sampler.

        Args:
            config (EasyDict): the config object to use
        """
        self.config = config

    @abc.abstractmethod
    def sample(self) -> None:
        """Sample from the model. Loads the checkpoint, load the modes, generates samples, evaluates, saves and plot them."""
        pass


class Sampler_Graph(Sampler):
    """Sampler for generic graph generation tasks"""

    def __init__(self, config: EasyDict) -> None:
        """Initialize the sampler with the config and the device.

        Args:
            config (EasyDict): the config object to use
        """
        super(Sampler_Graph, self).__init__(config)

        self.config = config
        self.device = load_device()
        self.device0 = self.device[0] if isinstance(self.device, list) else self.device
        self.n_sample = None

    def __repr__(self) -> str:
        """Return the string representation of the sampler."""
        return self.__class__.__name__

    def sample(self) -> None:
        """Sample from the model. Loads the checkpoint, load the modes, generates samples, evaluates, saves and plot them."""
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device, is_cc=self.is_cc)
        self.configt = self.ckpt_dict["config"]

        load_seed(self.configt.seed)
        self.train_graph_list, self.test_graph_list = load_data(
            self.configt, get_list=True
        )

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.config_name}_{self.config.ckpt}-sample_{self.config.current_time}"
        logger = Logger(
            str(os.path.join(self.log_dir, f"{self.log_name}.log")), mode="a"
        )

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f"{self.log_name}")
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model_x = load_model_from_ckpt(
            self.ckpt_dict["params_x"], self.ckpt_dict["x_state_dict"], self.device
        )
        self.model_adj = load_model_from_ckpt(
            self.ckpt_dict["params_adj"], self.ckpt_dict["adj_state_dict"], self.device
        )

        if self.config.sample.use_ema:
            self.ema_x = load_ema_from_ckpt(
                self.model_x, self.ckpt_dict["ema_x"], self.configt.train.ema
            )
            self.ema_adj = load_ema_from_ckpt(
                self.model_adj, self.ckpt_dict["ema_adj"], self.configt.train.ema
            )

            self.ema_x.copy_to(self.model_x.parameters())
            self.ema_adj.copy_to(self.model_adj.parameters())

        self.sampling_fn = load_sampling_fn(
            self.configt, self.config.sampler, self.config.sample, self.device
        )

        # -------- Generate samples --------
        logger.log(f"GEN SEED: {self.config.sample.seed}")
        load_seed(self.config.sample.seed)

        num_sampling_rounds = math.ceil(
            len(self.test_graph_list) / self.configt.data.batch_size
        )
        gen_graph_list = []
        for r in range(num_sampling_rounds):
            t_start = time.time()

            self.init_flags = init_flags(
                self.train_graph_list, self.configt, self.n_sample
            ).to(self.device0)

            x, adj, _ = self.sampling_fn(self.model_x, self.model_adj, self.init_flags)

            logger.log(f"Round {r} : {time.time()-t_start:.2f}s")

            samples_int = quantize(adj)
            gen_graph_list.extend(adjs_to_graphs(samples_int, True))

        gen_graph_list = gen_graph_list[: len(self.test_graph_list)]

        # -------- Evaluation --------
        methods, kernels = load_eval_settings(self.config.data.data)
        result_dict = eval_graph_list(
            self.test_graph_list, gen_graph_list, methods=methods, kernels=kernels
        )
        logger.log(f"MMD_full {result_dict}", verbose=False)
        logger.log(100 * "=")

        # -------- Save samples & Plot --------
        # Graphs
        save_dir = save_graph_list(
            self.log_folder_name, self.log_name + "_graphs", gen_graph_list
        )
        with open(save_dir, "rb") as f:
            sample_graph_list = pickle.load(f)
        plot_graphs_list(
            graphs=sample_graph_list,
            title=f"{self.config.ckpt}_graphs",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(*["samples", "fig", self.log_folder_name]),
                f"{self.config.ckpt}_graphs.png",
            )
            wandb.log({"Generated Graphs": wandb.Image(img_path)})


class Sampler_CC(Sampler):
    """Sampler for generic combinatorial complexes generation tasks"""

    def __init__(self, config: EasyDict) -> None:
        """Initialize the sampler with the config and the device.

        Args:
            config (EasyDict): the config object to use
        """
        super(Sampler_CC, self).__init__(config)

        self.config = config
        self.device = load_device()
        self.device0 = self.device[0] if isinstance(self.device, list) else self.device
        self.n_sample = None

    def __repr__(self) -> str:
        """Return the string representation of the sampler."""
        return self.__class__.__name__

    def sample(self) -> None:
        """Sample from the model. Loads the checkpoint, load the modes, generates samples, evaluates, saves and plot them."""
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device, is_cc=True)
        self.configt = self.ckpt_dict["config"]

        load_seed(self.configt.seed)
        self.train_CC_list, self.test_CC_list = load_data(
            self.configt, get_list=True, is_cc=True
        )

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.config_name}_{self.config.ckpt}-sample_{self.config.current_time}"
        logger = Logger(
            str(os.path.join(self.log_dir, f"{self.log_name}.log")), mode="a"
        )

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f"{self.log_name}")
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model_x = load_model_from_ckpt(
            self.ckpt_dict["params_x"], self.ckpt_dict["x_state_dict"], self.device
        )
        self.model_adj = load_model_from_ckpt(
            self.ckpt_dict["params_adj"], self.ckpt_dict["adj_state_dict"], self.device
        )
        self.model_rank2 = load_model_from_ckpt(
            self.ckpt_dict["params_rank2"],
            self.ckpt_dict["rank2_state_dict"],
            self.device,
        )

        if self.config.sample.use_ema:
            self.ema_x = load_ema_from_ckpt(
                self.model_x, self.ckpt_dict["ema_x"], self.configt.train.ema
            )
            self.ema_adj = load_ema_from_ckpt(
                self.model_adj, self.ckpt_dict["ema_adj"], self.configt.train.ema
            )
            self.ema_rank2 = load_ema_from_ckpt(
                self.model_rank2, self.ckpt_dict["ema_rank2"], self.configt.train.ema
            )

            self.ema_x.copy_to(self.model_x.parameters())
            self.ema_adj.copy_to(self.model_adj.parameters())
            self.ema_rank2.copy_to(self.model_rank2.parameters())

        self.sampling_fn = load_sampling_fn(
            self.configt,
            self.config.sampler,
            self.config.sample,
            self.device,
            is_cc=True,
            d_min=self.config.data.d_min,
            d_max=self.config.data.d_max,
        )

        # -------- Generate samples --------
        logger.log(f"GEN SEED: {self.config.sample.seed}")
        load_seed(self.config.sample.seed)

        num_sampling_rounds = math.ceil(
            len(self.test_CC_list) / self.configt.data.batch_size
        )
        gen_CC_list = []
        for r in range(num_sampling_rounds):
            t_start = time.time()

            self.init_flags = init_flags(
                self.train_CC_list, self.configt, self.n_sample, is_cc=True
            ).to(self.device0)

            x, adj, rank2, _ = self.sampling_fn(
                self.model_x, self.model_adj, self.model_rank2, self.init_flags
            )

            logger.log(f"Round {r} : {time.time()-t_start:.2f}s")

            samples_int = quantize(adj)
            sample_int_rank2 = quantize(rank2)
            gen_CC_list.extend(
                [
                    cc_from_incidence(x_, adj_, rank2_)
                    for x_, adj_, rank2_ in zip(x, samples_int, sample_int_rank2)
                ]
            )

        gen_CC_list = gen_CC_list[: len(self.test_CC_list)]

        # -------- Evaluation --------
        # Convert CC into graphs for evaluation
        self.test_graph_list = convert_CC_to_graphs(self.test_CC_list)
        gen_graph_list = convert_CC_to_graphs(gen_CC_list)

        methods, kernels = load_eval_settings(self.config.data.data)
        result_dict = eval_graph_list(
            self.test_graph_list, gen_graph_list, methods=methods, kernels=kernels
        )
        logger.log(f"MMD_full {result_dict}", verbose=False)
        logger.log(100 * "=")

        # -------- Save samples & Plot --------
        # Ccs
        save_dir = save_cc_list(
            self.log_folder_name, self.log_name + "_ccs", gen_CC_list
        )
        with open(save_dir, "rb") as f:
            sample_CC_list = pickle.load(f)
        plot_cc_list(
            ccs=sample_CC_list,
            title=f"{self.config.ckpt}_ccs",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(*["samples", "fig", self.log_folder_name]),
                f"{self.config.ckpt}_ccs.png",
            )
            wandb.log({"Generated Combinatorial Complexes": wandb.Image(img_path)})
        # Graphs
        save_dir = save_graph_list(
            self.log_folder_name, self.log_name + "_graphs", gen_graph_list
        )
        with open(save_dir, "rb") as f:
            sample_graph_list = pickle.load(f)
        plot_graphs_list(
            graphs=sample_graph_list,
            title=f"{self.config.ckpt}_graphs",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(*["samples", "fig", self.log_folder_name]),
                f"{self.config.ckpt}_graphs.png",
            )
            wandb.log({"Generated Graphs": wandb.Image(img_path)})


class Sampler_mol_Graph(Sampler):
    """Sampler for molecule generation tasks"""

    def __init__(self, config: EasyDict) -> None:
        """Initialize the sampler with the config and the device.

        Args:
            config (EasyDict): the config object to use
        """
        super(Sampler_mol_Graph, self).__init__(config)

        self.config = config
        self.device = load_device()
        self.device0 = self.device[0] if isinstance(self.device, list) else self.device
        self.n_sample = self.config.sample.n_sample

    def __repr__(self) -> str:
        """Return the string representation of the sampler."""
        return self.__class__.__name__

    def sample(self) -> None:
        """Sample from the model. Loads the checkpoint, load the modes, generates samples, evaluates and saves them."""
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict["config"]

        load_seed(self.config.seed)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.config_name}_{self.config.ckpt}-sample_{self.config.current_time}"
        logger = Logger(
            str(os.path.join(self.log_dir, f"{self.log_name}.log")), mode="a"
        )

        if not check_log(self.log_folder_name, self.log_name):
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model_x = load_model_from_ckpt(
            self.ckpt_dict["params_x"], self.ckpt_dict["x_state_dict"], self.device
        )
        self.model_adj = load_model_from_ckpt(
            self.ckpt_dict["params_adj"], self.ckpt_dict["adj_state_dict"], self.device
        )

        self.sampling_fn = load_sampling_fn(
            self.configt, self.config.sampler, self.config.sample, self.device
        )

        # -------- Generate samples --------
        logger.log(f"GEN SEED: {self.config.sample.seed}")
        load_seed(self.config.sample.seed)

        train_smiles, test_smiles = load_smiles(self.configt.data.data)
        train_smiles, test_smiles = canonicalize_smiles(
            train_smiles
        ), canonicalize_smiles(test_smiles)

        self.train_graph_list, _ = load_data(
            self.configt, get_list=True
        )  # for init_flags
        with open(f"data/{self.configt.data.data.lower()}_test_nx.pkl", "rb") as f:
            self.test_graph_list = pickle.load(f)  # for NSPDK MMD

        self.init_flags = init_flags(
            self.train_graph_list, self.configt, self.n_sample
        ).to(self.device0)
        x, adj, _ = self.sampling_fn(self.model_x, self.model_adj, self.init_flags)

        samples_int = quantize_mol(adj)

        samples_int = samples_int - 1
        samples_int[samples_int == -1] = 3  # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2

        adj = torch.nn.functional.one_hot(
            torch.tensor(samples_int), num_classes=4
        ).permute(0, 3, 1, 2)
        x = torch.where(x > 0.5, 1, 0)
        x = torch.concat(
            [x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1
        )  # 32, 9, 4 -> 32, 9, 5

        gen_mols, num_mols_wo_correction = gen_mol(x, adj, self.configt.data.data)
        num_mols = len(gen_mols)

        gen_smiles = mols_to_smiles(gen_mols)
        gen_smiles = [smi for smi in gen_smiles if len(smi)]

        # Convert generated molecules into graphs
        gen_graph_list = mols_to_nx(gen_mols)

        # -------- Save generated molecules --------
        with open(os.path.join(self.log_dir, f"{self.log_name}.txt"), "a") as f:
            for smiles in gen_smiles:
                f.write(f"{smiles}\n")

        # -------- Evaluation --------
        scores = get_all_metrics(
            gen=gen_smiles,
            k=len(gen_smiles),
            device=self.device0,
            n_jobs=8,
            test=test_smiles,
            train=train_smiles,
        )
        scores_nspdk = eval_graph_list(
            self.test_graph_list, gen_graph_list, methods=["nspdk"]
        )["nspdk"]

        logger.log(f"Number of molecules: {num_mols}")
        logger.log(f"validity w/o correction: {num_mols_wo_correction / num_mols}")
        for metric in ["valid", f"unique@{len(gen_smiles)}", "FCD/Test", "Novelty"]:
            logger.log(f"{metric}: {scores[metric]}")
        logger.log(f"NSPDK MMD: {scores_nspdk}")
        logger.log(100 * "=")

        # -------- Save samples & Plot --------
        # Graphs
        save_dir = save_graph_list(
            self.log_folder_name, self.log_name + "_mol_graphs", gen_graph_list
        )
        with open(save_dir, "rb") as f:
            sample_graph_list = pickle.load(f)
        plot_graphs_list(
            graphs=sample_graph_list,
            title=f"{self.config.ckpt}_mol_graphs",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(*["samples", "fig", self.log_folder_name]),
                f"{self.config.ckpt}_mol_graphs.png",
            )
            wandb.log({"Generated Mol Graphs": wandb.Image(img_path)})
        # Molecules
        save_dir = save_molecule_list(
            self.log_folder_name, self.log_name + "_mols", gen_mols
        )
        with open(save_dir, "rb") as f:
            sample_mol_list = pickle.load(f)
        plot_molecule_list(
            mols=sample_mol_list,
            title=f"{self.config.ckpt}_mols",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(*["samples", "fig", self.log_folder_name]),
                f"{self.config.ckpt}_mols.png",
            )
            wandb.log({"Generated Molecules": wandb.Image(img_path)})
        # 3D Molecule
        molecule = gen_mols[0]
        mol_3d = plot_3D_molecule(molecule)
        filedir = os.path.join(*["samples", "fig", self.log_folder_name])
        filename = f"{self.config.ckpt}_mols_3d.gif"
        rotate_molecule_animation(
            mol_3d,
            filedir=filedir,
            filename=filename,
            duration=1.0,
            frames=30,
            rotations_per_sec=1.0,
            overwrite=True,
            engine="kaleido",
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(filedir, filename)
            wandb.log({"Generated Molecules 3D": wandb.Image(img_path)})


class Sampler_mol_CC(Sampler):
    """Sampler for molecule generation tasks with combinatorial complexes"""

    def __init__(self, config: EasyDict) -> None:
        """Initialize the sampler with the config and the device.

        Args:
            config (EasyDict): the config object to use
        """
        super(Sampler_mol_CC, self).__init__(config)

        self.config = config
        self.device = load_device()
        self.device0 = (
            self.device[0] if isinstance(self.device, list) else self.device
        )  #
        self.n_sample = self.config.sample.n_sample

    def __repr__(self) -> str:
        """Return the string representation of the sampler."""
        return self.__class__.__name__

    def sample(self) -> None:
        """Sample from the model. Loads the checkpoint, load the modes, generates samples, evaluates and saves them."""
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device, is_cc=True)
        self.configt = self.ckpt_dict["config"]

        load_seed(self.config.seed)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.config_name}_{self.config.ckpt}-sample_{self.config.current_time}"
        logger = Logger(
            str(os.path.join(self.log_dir, f"{self.log_name}.log")), mode="a"
        )

        if not check_log(self.log_folder_name, self.log_name):
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model_x = load_model_from_ckpt(
            self.ckpt_dict["params_x"], self.ckpt_dict["x_state_dict"], self.device
        )
        self.model_adj = load_model_from_ckpt(
            self.ckpt_dict["params_adj"], self.ckpt_dict["adj_state_dict"], self.device
        )
        self.model_rank2 = load_model_from_ckpt(
            self.ckpt_dict["params_rank2"],
            self.ckpt_dict["rank2_state_dict"],
            self.device,
        )

        self.sampling_fn = load_sampling_fn(
            self.configt,
            self.config.sampler,
            self.config.sample,
            self.device,
            is_cc=True,
            d_min=self.config.data.d_min,
            d_max=self.config.data.d_max,
        )

        # -------- Generate samples --------
        logger.log(f"GEN SEED: {self.config.sample.seed}")
        load_seed(self.config.sample.seed)

        train_smiles, test_smiles = load_smiles(self.configt.data.data)
        train_smiles, test_smiles = canonicalize_smiles(
            train_smiles
        ), canonicalize_smiles(test_smiles)

        self.train_CC_list, _ = load_data(
            self.configt, get_list=True, is_cc=True
        )  # for init_flags
        with open(f"data/{self.configt.data.data.lower()}_test_nx.pkl", "rb") as f:
            self.test_graph_list = pickle.load(f)  # for NSPDK MMD

        self.init_flags = init_flags(
            self.train_CC_list, self.configt, self.n_sample, is_cc=True
        ).to(self.device0)
        x, adj, rank2, _ = self.sampling_fn(
            self.model_x, self.model_adj, self.model_rank2, self.init_flags
        )

        samples_int = quantize_mol(adj)
        samples_int_rank2 = quantize(rank2)

        samples_int = samples_int - 1
        samples_int[samples_int == -1] = 3  # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2

        adj = torch.nn.functional.one_hot(
            torch.tensor(samples_int), num_classes=4
        ).permute(0, 3, 1, 2)
        x = torch.where(x > 0.5, 1, 0)
        x = torch.concat(
            [x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1
        )  # 32, 9, 4 -> 32, 9, 5

        gen_mols, num_mols_wo_correction = gen_mol(x, adj, self.configt.data.data)
        num_mols = len(gen_mols)

        gen_smiles = mols_to_smiles(gen_mols)
        gen_smiles = [smi for smi in gen_smiles if len(smi)]

        # Convert generated molecules into graphs and combinatorial complexes
        gen_graph_list = mols_to_nx(gen_mols)
        gen_CC_list = mols_to_cc(gen_mols)

        # -------- Save generated molecules --------
        with open(os.path.join(self.log_dir, f"{self.log_name}_smiles.txt"), "a") as f:
            for smiles in gen_smiles:
                f.write(f"{smiles}\n")

        # -------- Evaluation --------
        scores = get_all_metrics(
            gen=gen_smiles,
            k=len(gen_smiles),
            device=self.device0,
            n_jobs=8,
            test=test_smiles,
            train=train_smiles,
        )
        scores_nspdk = eval_graph_list(
            self.test_graph_list, gen_graph_list, methods=["nspdk"]
        )["nspdk"]

        logger.log(f"Number of molecules: {num_mols}")
        logger.log(f"validity w/o correction: {num_mols_wo_correction / num_mols}")
        for metric in ["valid", f"unique@{len(gen_smiles)}", "FCD/Test", "Novelty"]:
            logger.log(f"{metric}: {scores[metric]}")
        logger.log(f"NSPDK MMD: {scores_nspdk}")
        logger.log(100 * "=")

        # -------- Save samples & Plot --------
        # Ccs
        save_dir = save_cc_list(
            self.log_folder_name, self.log_name + "_mol_ccs", gen_CC_list
        )
        with open(save_dir, "rb") as f:
            sample_CC_list = pickle.load(f)
        plot_cc_list(
            ccs=sample_CC_list,
            title=f"{self.config.ckpt}_mol_ccs",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(*["samples", "fig", self.log_folder_name]),
                f"{self.config.ckpt}_mol_ccs.png",
            )
            wandb.log({"Generated Mol Combinatorial Complexes": wandb.Image(img_path)})
        # Graphs
        save_dir = save_graph_list(
            self.log_folder_name, self.log_name + "_mol_graphs", gen_graph_list
        )
        with open(save_dir, "rb") as f:
            sample_graph_list = pickle.load(f)
        plot_graphs_list(
            graphs=sample_graph_list,
            title=f"{self.config.ckpt}_mol_graphs",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(*["samples", "fig", self.log_folder_name]),
                f"{self.config.ckpt}_mol_graphs.png",
            )
            wandb.log({"Generated Mol Graphs": wandb.Image(img_path)})
        # Molecules
        save_dir = save_molecule_list(
            self.log_folder_name, self.log_name + "_mols", gen_mols
        )
        with open(save_dir, "rb") as f:
            sample_mol_list = pickle.load(f)
        plot_molecule_list(
            mols=sample_mol_list,
            title=f"{self.config.ckpt}_mols",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(*["samples", "fig", self.log_folder_name]),
                f"{self.config.ckpt}_mols.png",
            )
            wandb.log({"Generated Molecules": wandb.Image(img_path)})
        # 3D Molecule
        molecule = gen_mols[0]
        mol_3d = plot_3D_molecule(molecule)
        filedir = os.path.join(*["samples", "fig", self.log_folder_name])
        filename = f"{self.config.ckpt}_mols_3d.gif"
        rotate_molecule_animation(
            mol_3d,
            filedir=filedir,
            filename=filename,
            duration=1.0,
            frames=30,
            rotations_per_sec=1.0,
            overwrite=True,
            engine="kaleido",
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(filedir, filename)
            wandb.log({"Generated Molecule 3D": wandb.Image(img_path)})


def get_sampler_from_config(
    config: EasyDict,
) -> Sampler:
    if config.is_cc:
        sampler = (
            Sampler_mol_CC(config)
            if is_molecular_config(config)
            else Sampler_CC(config)
        )
    else:
        sampler = (
            Sampler_mol_Graph(config)
            if is_molecular_config(config)
            else Sampler_Graph(config)
        )
    return sampler

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""sampler.py: code for sampling from the model.
"""

import abc
import math
import os
import pickle
from time import perf_counter

import torch
import wandb
from easydict import EasyDict
from moses.metrics.metrics import get_all_metrics

from ccsd.src.evaluation.stats import eval_graph_list
from ccsd.src.utils.cc_utils import (
    cc_from_incidence,
    convert_CC_to_graphs,
    convert_graphs_to_CCs,
    eval_CC_list,
    init_flags,
    load_cc_eval_settings,
    mols_to_cc,
)
from ccsd.src.utils.graph_utils import (
    adjs_to_graphs,
    nxs_to_mols,
    quantize,
    quantize_mol,
)
from ccsd.src.utils.loader import (
    load_ckpt,
    load_data,
    load_device,
    load_ema_from_ckpt,
    load_eval_settings,
    load_model_from_ckpt,
    load_sampling_fn,
    load_seed,
)
from ccsd.src.utils.logger import (
    Logger,
    check_log,
    device_log,
    sample_log,
    set_log,
    start_log,
    time_log,
    train_log,
)
from ccsd.src.utils.mol_utils import (
    canonicalize_smiles,
    gen_mol,
    is_molecular_config,
    load_smiles,
    mols_to_nx,
    mols_to_smiles,
)
from ccsd.src.utils.plot import (
    diffusion_animation,
    plot_3D_molecule,
    plot_cc_list,
    plot_graphs_list,
    plot_molecule_list,
    rotate_molecule_animation,
    save_cc_list,
    save_graph_list,
    save_molecule_list,
)


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
    """Sampler for generic graph generation tasks

    Adapted from Jo, J. & al (2022)
    """

    def __init__(self, config: EasyDict) -> None:
        """Initialize the sampler with the config and the device.

        Args:
            config (EasyDict): the config object to use
        """
        super(Sampler_Graph, self).__init__(config)

        self.config = config
        self.device = load_device()
        self.device0 = self.device[0] if isinstance(self.device, list) else self.device
        # Device to compute the score metrics
        if self.device0 == "cpu":
            self.device_score = "cpu"
        elif "cuda" in str(self.device0):
            self.device_score = str(self.device0)
        else:
            self.device_score = f"cuda:{self.device0}"
        self.n_samples = None
        self.cc_nb_eval = None
        # Worker kwargs for CC eval
        self.worker_kwargs = {
            "min_node_val": self.config.data.min_node_val,
            "max_node_val": self.config.data.max_node_val,
            "node_label": self.config.data.node_label,
            "min_edge_val": self.config.data.min_edge_val,
            "max_edge_val": self.config.data.max_edge_val,
            "edge_label": self.config.data.edge_label,
            "d_min": self.config.data.d_min,
            "d_max": self.config.data.d_max,
            "N": self.config.data.max_node_num,
        }
        self.divide_batch = self.config.sample.get("divide_batch", 1)

    def __repr__(self) -> str:
        """Return the string representation of the sampler."""
        return f"{self.__class__.__name__}(batch_size={self.config.data.batch_size}, cc_nb_eval={self.cc_nb_eval})"

    def sample(self) -> None:
        """Sample from the model. Loads the checkpoint, load the modes, generates samples, evaluates, saves and plot them."""
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device, is_cc=False)
        self.configt = self.ckpt_dict["config"]

        load_seed(self.configt.seed)
        self.train_graph_list, self.test_graph_list = load_data(
            self.configt, get_list=True
        )

        self.log_folder_name, self.log_dir, _ = set_log(
            self.configt, is_train=False, folder=self.config.folder
        )
        self.log_name = f"{self.config.config_name}_{self.config.ckpt}-sample_{self.config.current_time}"
        logger = Logger(
            str(os.path.join(self.log_dir, f"{self.log_name}.log")), mode="a"
        )

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f"{self.log_name}")
            device_log(logger, self.device)
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

        for m in [self.model_x, self.model_adj]:
            logger.log(
                f"Model {m.__class__.__name__} loaded on {next(m.parameters()).device.type}"
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
            self.configt,
            self.config.sampler,
            self.config.sample,
            self.device,
            divide_batch=self.divide_batch,
        )

        # -------- Generate samples --------
        logger.log(f"GEN SEED: {self.config.sample.seed}")
        load_seed(self.config.sample.seed)

        num_sampling_rounds = math.ceil(
            len(self.test_graph_list) / self.configt.data.batch_size
        )
        gen_graph_list = []
        logger.log(
            f"Number sampling rounds: {num_sampling_rounds}, number of samples per round: {self.config.data.batch_size}"
        )
        start_sampling_time = perf_counter()
        for r in range(num_sampling_rounds):
            t_start = perf_counter()

            self.init_flags = init_flags(
                self.train_graph_list, self.configt, self.n_samples // self.divide_batch
            ).to(self.device0)

            x, adj, _, diff_traj = self.sampling_fn(
                self.model_x, self.model_adj, self.init_flags
            )

            for _ in range(1, self.divide_batch):
                x_, adj_, _, _ = self.sampling_fn(
                    self.model_x, self.model_adj, self.init_flags
                )
                x = torch.cat((x, x_), dim=0)
                adj = torch.cat((adj, adj_), dim=0)

            logger.log(f"Round {r} : {perf_counter()-t_start:.2f}s")

            samples_int = quantize(adj)
            gen_graph_list.extend(adjs_to_graphs(samples_int, True))

        gen_graph_list = gen_graph_list[: len(self.test_graph_list)]
        print("Sampling done.")
        sampling_time = perf_counter() - start_sampling_time
        time_log(logger, time_type="sample", elapsed_time=sampling_time)
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            wandb.log({"Sampling time": sampling_time})

        # -------- Evaluation --------
        # Eval graphs
        methods, kernels = load_eval_settings(self.config.data.data)
        result_dict_graph = eval_graph_list(
            self.test_graph_list,
            gen_graph_list,
            methods=methods,
            kernels=kernels,
            folder=self.config.folder,
        )

        # Eval lifted CCs from the graphs

        # Lift the test graphs and generate graphs into CCs for evaluation
        lifting_procedure = self.config.data.lifting_procedure
        lifting_procedure_kwargs = self.config.data.lifting_procedure_kwargs
        self.test_CC_list = convert_graphs_to_CCs(
            self.test_graph_list,
            is_molecule=False,
            lifting_procedure=lifting_procedure,
            lifting_procedure_kwargs=lifting_procedure_kwargs,
            max_nb_nodes=self.config.data.max_node_num,
        )
        gen_CC_list = convert_graphs_to_CCs(
            gen_graph_list,
            is_molecule=False,
            lifting_procedure=lifting_procedure,
            lifting_procedure_kwargs=lifting_procedure_kwargs,
            max_nb_nodes=self.config.data.max_node_num,
        )  # same for the generated graphs

        methods, kernels = load_cc_eval_settings()
        result_dict_CC = eval_CC_list(
            self.test_CC_list,
            gen_CC_list,
            worker_kwargs=self.worker_kwargs,
            methods=methods,
            kernels=kernels,
            cc_nb_eval=self.cc_nb_eval,
        )

        logger.log(
            f"CCs Eval @{self.cc_nb_eval} {result_dict_CC}", verbose=False
        )  # verbose=False cause already printed

        logger.log(
            f"MMD_full {result_dict_graph}", verbose=False
        )  # verbose=False cause already printed
        logger.log(100 * "=")
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add scores to wandb
            wandb.log(result_dict_graph)

        # -------- Save samples & Plot --------
        # Graphs
        save_dir = save_graph_list(
            self.config, self.log_folder_name, self.log_name + "_graphs", gen_graph_list
        )
        with open(save_dir, "rb") as f:
            sample_graph_list = pickle.load(f)
        plot_graphs_list(
            config=self.config,
            graphs=sample_graph_list,
            title=f"graphs_{self.log_name}",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(
                    *[self.config.folder, "samples", "fig", self.log_folder_name]
                ),
                f"graphs_{self.log_name}.png",
            )
            wandb.log({"Generated Graphs": wandb.Image(img_path)})
        # Diffusion trajectory animation
        if self.config.general_config.plotly_fig:
            filedir = os.path.join(
                *[self.config.folder, "samples", "fig", self.log_folder_name]
            )
            filename = f"diff_traj_graphs_{self.log_name}.gif"
            diffusion_animation(
                diff_traj=diff_traj,
                is_molecule=False,
                filedir=filedir,
                filename=filename,
                fps=25,
                overwrite=True,
                engine=self.config.general_config.engine,
            )
            if (
                self.config.experiment_type == "train"
            ) and self.config.general_config.use_wandb:
                # add plots to wandb
                img_path = os.path.join(filedir, filename)
                wandb.log({"Diffusion Trajectory Graph": wandb.Image(img_path)})
            # Cropped
            filename = f"diff_traj_graphs_cropped_{self.log_name}.gif"
            diffusion_animation(
                diff_traj=diff_traj,
                is_molecule=False,
                filedir=filedir,
                filename=filename,
                fps=25,
                overwrite=True,
                engine=self.config.general_config.engine,
                cropped=True,
            )
            if (
                self.config.experiment_type == "train"
            ) and self.config.general_config.use_wandb:
                # add plots to wandb
                img_path = os.path.join(filedir, filename)
                wandb.log({"Diffusion Trajectory Graph Cropped": wandb.Image(img_path)})


class Sampler_CC(Sampler):
    """Sampler for generic combinatorial complexes generation tasks

    Adapted from Jo, J. & al (2022)
    """

    def __init__(self, config: EasyDict) -> None:
        """Initialize the sampler with the config and the device.

        Args:
            config (EasyDict): the config object to use
        """
        super(Sampler_CC, self).__init__(config)

        self.config = config
        self.device = load_device()
        self.device0 = self.device[0] if isinstance(self.device, list) else self.device
        # Device to compute the score metrics
        if self.device0 == "cpu":
            self.device_score = "cpu"
        elif "cuda" in str(self.device0):
            self.device_score = str(self.device0)
        else:
            self.device_score = f"cuda:{self.device0}"
        self.n_samples = None
        self.cc_nb_eval = None
        # Worker kwargs for CC eval
        self.worker_kwargs = {
            "min_node_val": self.config.data.min_node_val,
            "max_node_val": self.config.data.max_node_val,
            "node_label": self.config.data.node_label,
            "min_edge_val": self.config.data.min_edge_val,
            "max_edge_val": self.config.data.max_edge_val,
            "edge_label": self.config.data.edge_label,
            "d_min": self.config.data.d_min,
            "d_max": self.config.data.d_max,
            "N": self.config.data.max_node_num,
        }
        self.divide_batch = self.config.sample.get("divide_batch", 1)

    def __repr__(self) -> str:
        """Return the string representation of the sampler."""
        return f"{self.__class__.__name__}(batch_size={self.config.data.batch_size})"

    def sample(self) -> None:
        """Sample from the model. Loads the checkpoint, load the modes, generates samples, evaluates, saves and plot them."""
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device, is_cc=True)
        self.configt = self.ckpt_dict["config"]

        load_seed(self.configt.seed)
        self.train_CC_list, self.test_CC_list = load_data(
            self.configt, get_list=True, is_cc=True
        )

        self.log_folder_name, self.log_dir, _ = set_log(
            self.configt, is_train=False, folder=self.config.folder
        )
        self.log_name = f"{self.config.config_name}_{self.config.ckpt}-sample_{self.config.current_time}"
        logger = Logger(
            str(os.path.join(self.log_dir, f"{self.log_name}.log")), mode="a"
        )

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f"{self.log_name}")
            device_log(logger, self.device)
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
        for m in [self.model_x, self.model_adj, self.model_rank2]:
            logger.log(
                f"Model {m.__class__.__name__} loaded on {next(m.parameters()).device.type}"
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
            divide_batch=self.divide_batch,
        )

        # -------- Generate samples --------
        logger.log(f"GEN SEED: {self.config.sample.seed}")
        load_seed(self.config.sample.seed)

        num_sampling_rounds = math.ceil(
            len(self.test_CC_list) / self.configt.data.batch_size
        )
        gen_CC_list = []
        logger.log(
            f"Number sampling rounds: {num_sampling_rounds}, number of samples per round: {self.config.data.batch_size}"
        )
        start_sampling_time = perf_counter()
        for r in range(num_sampling_rounds):
            t_start = perf_counter()

            self.init_flags = init_flags(
                self.train_CC_list,
                self.configt,
                self.n_samples // self.divide_batch,
                is_cc=True,
            ).to(self.device0)

            x, adj, rank2, _, diff_traj = self.sampling_fn(
                self.model_x, self.model_adj, self.model_rank2, self.init_flags
            )

            for _ in range(1, self.divide_batch):
                x_, adj_, rank2_, _, _ = self.sampling_fn(
                    self.model_x, self.model_adj, self.model_rank2, self.init_flags
                )
                x = torch.cat((x, x_), dim=0)
                adj = torch.cat((adj, adj_), dim=0)
                rank2 = torch.cat((rank2, rank2_), dim=0)

            logger.log(f"Round {r} : {perf_counter()-t_start:.2f}s")

            samples_int = quantize(adj)
            sample_int_rank2 = quantize(rank2)
            gen_CC_list.extend(
                [
                    cc_from_incidence(
                        [x_, adj_, rank2_],
                        d_min=self.config.data.d_min,
                        d_max=self.config.data.d_max,
                        is_molecule=False,
                    )
                    for x_, adj_, rank2_ in zip(x, samples_int, sample_int_rank2)
                ]
            )

        gen_CC_list = gen_CC_list[: len(self.test_CC_list)]
        print("Sampling done.")
        sampling_time = perf_counter() - start_sampling_time
        time_log(logger, time_type="sample", elapsed_time=sampling_time)
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            wandb.log({"Sampling time": sampling_time})

        # -------- Evaluation --------
        # Convert CC into graphs for evaluation
        self.test_graph_list = convert_CC_to_graphs(self.test_CC_list)
        gen_graph_list = convert_CC_to_graphs(gen_CC_list)

        # Eval graphs
        methods, kernels = load_eval_settings(self.config.data.data)
        result_dict_graph = eval_graph_list(
            self.test_graph_list,
            gen_graph_list,
            methods=methods,
            kernels=kernels,
            folder=self.config.folder,
        )

        # Eval CCs
        methods, kernels = load_cc_eval_settings()
        result_dict_CC = eval_CC_list(
            self.test_CC_list,
            gen_CC_list,
            worker_kwargs=self.worker_kwargs,
            methods=methods,
            kernels=kernels,
            cc_nb_eval=self.cc_nb_eval,
        )

        logger.log(
            f"MMD_full {result_dict_graph}", verbose=False
        )  # verbose=False cause already printed
        logger.log(
            f"CCs eval @{self.cc_nb_eval} {result_dict_CC}", verbose=False
        )  # verbose=False cause already printed
        logger.log(100 * "=")
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add scores to wandb
            wandb.log(result_dict_graph)
            wandb.log(result_dict_CC)

        # -------- Save samples & Plot --------
        # Ccs
        save_dir = save_cc_list(
            self.config, self.log_folder_name, self.log_name + "_ccs", gen_CC_list
        )
        with open(save_dir, "rb") as f:
            sample_CC_list = pickle.load(f)
        plot_cc_list(
            config=self.config,
            ccs=sample_CC_list,
            title=f"ccs_{self.log_name}",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(
                    *[self.config.folder, "samples", "fig", self.log_folder_name]
                ),
                f"ccs_{self.log_name}.png",
            )
            wandb.log({"Generated Combinatorial Complexes": wandb.Image(img_path)})
        # Graphs
        save_dir = save_graph_list(
            self.config, self.log_folder_name, self.log_name + "_graphs", gen_graph_list
        )
        with open(save_dir, "rb") as f:
            sample_graph_list = pickle.load(f)
        plot_graphs_list(
            config=self.config,
            graphs=sample_graph_list,
            title=f"graphs_{self.log_name}",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(
                    *[self.config.folder, "samples", "fig", self.log_folder_name]
                ),
                f"graphs_{self.log_name}.png",
            )
            wandb.log({"Generated Graphs": wandb.Image(img_path)})
        # Diffusion trajectory animation
        if self.config.general_config.plotly_fig:
            filedir = os.path.join(
                *[self.config.folder, "samples", "fig", self.log_folder_name]
            )
            filename = f"diff_traj_graphs_{self.log_name}.gif"
            diffusion_animation(
                diff_traj=diff_traj,
                is_molecule=False,
                filedir=filedir,
                filename=filename,
                fps=25,
                overwrite=True,
                engine=self.config.general_config.engine,
            )
            if (
                self.config.experiment_type == "train"
            ) and self.config.general_config.use_wandb:
                # add plots to wandb
                img_path = os.path.join(filedir, filename)
                wandb.log({"Diffusion Trajectory Graph": wandb.Image(img_path)})
            # Cropped
            filename = f"diff_traj_graphs_cropped_{self.log_name}.gif"
            diffusion_animation(
                diff_traj=diff_traj,
                is_molecule=False,
                filedir=filedir,
                filename=filename,
                fps=25,
                overwrite=True,
                engine=self.config.general_config.engine,
                cropped=True,
            )
            if (
                self.config.experiment_type == "train"
            ) and self.config.general_config.use_wandb:
                # add plots to wandb
                img_path = os.path.join(filedir, filename)
                wandb.log({"Diffusion Trajectory Graph Cropped": wandb.Image(img_path)})


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
        # Device to compute the score metrics
        if self.device0 == "cpu":
            self.device_score = "cpu"
        elif "cuda" in str(self.device0):
            self.device_score = str(self.device0)
        else:
            self.device_score = f"cuda:{self.device0}"
        self.n_samples = self.config.sample.n_samples
        self.cc_nb_eval = self.config.sample.cc_nb_eval
        # Worker kwargs for CC eval
        self.worker_kwargs = {
            "min_node_val": self.config.data.min_node_val,
            "max_node_val": self.config.data.max_node_val,
            "node_label": self.config.data.node_label,
            "min_edge_val": self.config.data.min_edge_val,
            "max_edge_val": self.config.data.max_edge_val,
            "edge_label": self.config.data.edge_label,
            "d_min": self.config.data.d_min,
            "d_max": self.config.data.d_max,
            "N": self.config.data.max_node_num,
        }
        self.divide_batch = self.config.sample.get("divide_batch", 1)

    def __repr__(self) -> str:
        """Return the string representation of the sampler."""
        return f"{self.__class__.__name__}(n_samples={self.n_samples}, cc_nb_eval={self.cc_nb_eval})"

    def sample(self) -> None:
        """Sample from the model. Loads the checkpoint, load the modes, generates samples, evaluates and saves them."""
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict["config"]

        load_seed(self.config.seed)

        self.log_folder_name, self.log_dir, _ = set_log(
            self.configt, is_train=False, folder=self.config.folder
        )
        self.log_name = f"{self.config.config_name}_{self.config.ckpt}-sample_{self.config.current_time}"
        logger = Logger(
            str(os.path.join(self.log_dir, f"{self.log_name}.log")), mode="a"
        )

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f"{self.log_name}")
            device_log(logger, self.device)
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
        for m in [self.model_x, self.model_adj]:
            logger.log(
                f"Model {m.__class__.__name__} loaded on {next(m.parameters()).device.type}"
            )

        self.sampling_fn = load_sampling_fn(
            self.configt,
            self.config.sampler,
            self.config.sample,
            self.device,
            divide_batch=self.divide_batch,
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
        with open(
            os.path.join(
                self.config.folder,
                "data",
                f"{self.configt.data.data.lower()}_test_nx.pkl",
            ),
            "rb",
        ) as f:
            self.test_graph_list = pickle.load(f)  # for NSPDK MMD

        logger.log(f"Sampling {self.n_samples} samples ...")
        start_sampling_time = perf_counter()
        self.init_flags = init_flags(
            self.train_graph_list, self.configt, self.n_samples // self.divide_batch
        ).to(self.device0)
        x, adj, _, diff_traj = self.sampling_fn(
            self.model_x, self.model_adj, self.init_flags
        )

        for _ in range(1, self.divide_batch):
            x_, adj_, _, _ = self.sampling_fn(
                self.model_x, self.model_adj, self.init_flags
            )
            x = torch.cat((x, x_), dim=0)
            adj = torch.cat((adj, adj_), dim=0)

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
        print("Sampling done.")
        sampling_time = perf_counter() - start_sampling_time
        time_log(logger, time_type="sample", elapsed_time=sampling_time)
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            wandb.log({"Sampling time": sampling_time})

        # -------- Save generated molecules --------
        with open(os.path.join(self.log_dir, f"{self.log_name}.txt"), "a") as f:
            for smiles in gen_smiles:
                f.write(f"{smiles}\n")

        # -------- Evaluation --------
        # Eval molecules
        scores = get_all_metrics(
            gen=gen_smiles,
            k=len(gen_smiles),
            device=self.device_score,
            n_jobs=8,
            test=test_smiles,
            train=train_smiles,
        )
        scores_nspdk = eval_graph_list(
            self.test_graph_list,
            gen_graph_list,
            methods=["nspdk"],
            folder=self.config.folder,
        )["nspdk"]

        # Eval lifted CCs from the graphs

        # Create test_CC_list based on test_graph_list via a conversion to molecules
        test_mol_list = nxs_to_mols(self.test_graph_list)
        self.test_CC_list = mols_to_cc(test_mol_list)
        gen_CC_list = mols_to_cc(gen_mols)  # same for the generated molecules

        methods, kernels = load_cc_eval_settings()
        result_dict_CC = eval_CC_list(
            self.test_CC_list,
            gen_CC_list,
            worker_kwargs=self.worker_kwargs,
            methods=methods,
            kernels=kernels,
            cc_nb_eval=self.cc_nb_eval,
        )

        logger.log(
            f"CCs Eval @{self.cc_nb_eval} {result_dict_CC}", verbose=False
        )  # verbose=False cause already printed

        logger.log(f"Number of molecules: {num_mols}")
        logger.log(f"validity w/o correction: {num_mols_wo_correction / num_mols}")
        for metric in ["valid", f"unique@{len(gen_smiles)}", "FCD/Test", "Novelty"]:
            logger.log(f"{metric}: {scores[metric]}")
        logger.log(f"NSPDK MMD: {scores_nspdk}")
        logger.log(100 * "=")
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add scores to wandb
            wandb.log(
                {
                    "validity": scores["valid"],
                    f"unique@{len(gen_smiles)}": scores[f"unique@{len(gen_smiles)}"],
                    "FCD/Test": scores["FCD/Test"],
                    "Novelty": scores["Novelty"],
                    "NSPDK MMD": scores_nspdk,
                }
            )

        # -------- Save samples & Plot --------
        # Graphs
        save_dir = save_graph_list(
            self.config,
            self.log_folder_name,
            self.log_name + "_mol_graphs",
            gen_graph_list,
        )
        with open(save_dir, "rb") as f:
            sample_graph_list = pickle.load(f)
        plot_graphs_list(
            config=self.config,
            graphs=sample_graph_list,
            title=f"mol_graphs_{self.log_name}",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(
                    *[self.config.folder, "samples", "fig", self.log_folder_name]
                ),
                f"mol_graphs_{self.log_name}.png",
            )
            wandb.log({"Generated Mol Graphs": wandb.Image(img_path)})
        # Molecules
        save_dir = save_molecule_list(
            self.config, self.log_folder_name, self.log_name + "_mols", gen_mols
        )
        with open(save_dir, "rb") as f:
            sample_mol_list = pickle.load(f)
        plot_molecule_list(
            config=self.config,
            mols=sample_mol_list,
            title=f"mols_{self.log_name}",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(
                    *[self.config.folder, "samples", "fig", self.log_folder_name]
                ),
                f"mols_{self.log_name}.png",
            )
            wandb.log({"Generated Molecules": wandb.Image(img_path)})
        # 3D Molecule
        if self.config.general_config.plotly_fig:
            molecule = gen_mols[0]
            mol_3d = plot_3D_molecule(molecule)
            filedir = os.path.join(
                *[self.config.folder, "samples", "fig", self.log_folder_name]
            )
            filename = f"mols_3d_{self.log_name}.gif"
            rotate_molecule_animation(
                mol_3d,
                filedir=filedir,
                filename=filename,
                duration=1.0,
                frames=30,
                rotations_per_sec=1.0,
                overwrite=True,
                engine=self.config.general_config.engine,
            )
            if (
                self.config.experiment_type == "train"
            ) and self.config.general_config.use_wandb:
                # add plots to wandb
                img_path = os.path.join(filedir, filename)
                wandb.log({"Generated Molecules 3D": wandb.Image(img_path)})
        # Diffusion trajectory animation - Graphs and molecules
        if self.config.general_config.plotly_fig:
            # Graph
            filedir = os.path.join(
                *[self.config.folder, "samples", "fig", self.log_folder_name]
            )
            filename = f"diff_traj_graphs_{self.log_name}.gif"
            diffusion_animation(
                diff_traj=diff_traj,
                is_molecule=False,
                filedir=filedir,
                filename=filename,
                fps=25,
                overwrite=True,
                engine=self.config.general_config.engine,
            )
            if (
                self.config.experiment_type == "train"
            ) and self.config.general_config.use_wandb:
                # add plots to wandb
                img_path = os.path.join(filedir, filename)
                wandb.log({"Diffusion Trajectory Graph": wandb.Image(img_path)})
            # Graph Cropped
            filename = f"diff_traj_graphs_cropped_{self.log_name}.gif"
            diffusion_animation(
                diff_traj=diff_traj,
                is_molecule=False,
                filedir=filedir,
                filename=filename,
                fps=25,
                overwrite=True,
                engine=self.config.general_config.engine,
                cropped=True,
            )
            if (
                self.config.experiment_type == "train"
            ) and self.config.general_config.use_wandb:
                # add plots to wandb
                img_path = os.path.join(filedir, filename)
                wandb.log({"Diffusion Trajectory Graph Cropped": wandb.Image(img_path)})
            # Molecule
            filename = f"diff_traj_mol_{self.log_name}.gif"
            diffusion_animation(
                diff_traj=diff_traj,
                is_molecule=True,
                filedir=filedir,
                filename=filename,
                fps=25,
                overwrite=True,
                engine=self.config.general_config.engine,
            )
            if (
                self.config.experiment_type == "train"
            ) and self.config.general_config.use_wandb:
                # add plots to wandb
                img_path = os.path.join(filedir, filename)
                wandb.log({"Diffusion Trajectory Molecule": wandb.Image(img_path)})
            # Molecule Cropped
            filename = f"diff_traj_mol_cropped_{self.log_name}.gif"
            diffusion_animation(
                diff_traj=diff_traj,
                is_molecule=True,
                filedir=filedir,
                filename=filename,
                fps=25,
                overwrite=True,
                engine=self.config.general_config.engine,
                cropped=True,
            )
            if (
                self.config.experiment_type == "train"
            ) and self.config.general_config.use_wandb:
                # add plots to wandb
                img_path = os.path.join(filedir, filename)
                wandb.log(
                    {"Diffusion Trajectory Molecule Cropped": wandb.Image(img_path)}
                )


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
        self.device0 = self.device[0] if isinstance(self.device, list) else self.device
        # Device to compute the score metrics
        if self.device0 == "cpu":
            self.device_score = "cpu"
        elif "cuda" in str(self.device0):
            self.device_score = str(self.device0)
        else:
            self.device_score = f"cuda:{self.device0}"
        self.n_samples = self.config.sample.n_samples
        self.cc_nb_eval = self.config.sample.cc_nb_eval
        # Worker kwargs for CC eval
        self.worker_kwargs = {
            "min_node_val": self.config.data.min_node_val,
            "max_node_val": self.config.data.max_node_val,
            "node_label": self.config.data.node_label,
            "min_edge_val": self.config.data.min_edge_val,
            "max_edge_val": self.config.data.max_edge_val,
            "edge_label": self.config.data.edge_label,
            "d_min": self.config.data.d_min,
            "d_max": self.config.data.d_max,
            "N": self.config.data.max_node_num,
        }
        self.divide_batch = self.config.sample.get("divide_batch", 1)

    def __repr__(self) -> str:
        """Return the string representation of the sampler."""
        return f"{self.__class__.__name__}(n_samples={self.n_samples}, cc_nb_eval={self.cc_nb_eval})"

    def sample(self) -> None:
        """Sample from the model. Loads the checkpoint, load the modes, generates samples, evaluates and saves them."""
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device, is_cc=True)
        self.configt = self.ckpt_dict["config"]

        load_seed(self.config.seed)

        self.log_folder_name, self.log_dir, _ = set_log(
            self.configt, is_train=False, folder=self.config.folder
        )
        self.log_name = f"{self.config.config_name}_{self.config.ckpt}-sample_{self.config.current_time}"
        logger = Logger(
            str(os.path.join(self.log_dir, f"{self.log_name}.log")), mode="a"
        )

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f"{self.log_name}")
            device_log(logger, self.device)
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
        for m in [self.model_x, self.model_adj, self.model_rank2]:
            logger.log(
                f"Model {m.__class__.__name__} loaded on {next(m.parameters()).device.type}"
            )

        self.sampling_fn = load_sampling_fn(
            self.configt,
            self.config.sampler,
            self.config.sample,
            self.device,
            is_cc=True,
            d_min=self.config.data.d_min,
            d_max=self.config.data.d_max,
            divide_batch=self.divide_batch,
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
        with open(
            os.path.join(
                self.config.folder,
                "data",
                f"{self.configt.data.data.lower()}_test_nx.pkl",
            ),
            "rb",
        ) as f:
            self.test_graph_list = pickle.load(f)  # for NSPDK MMD

        # Create test_CC_list based on test_graph_list via a conversion to molecules
        test_mol_list = nxs_to_mols(self.test_graph_list)
        self.test_CC_list = mols_to_cc(test_mol_list)

        # Generate samples
        logger.log(f"Sampling {self.n_samples} samples ...")
        start_sampling_time = perf_counter()
        self.init_flags = init_flags(
            self.train_CC_list,
            self.configt,
            self.n_samples // self.divide_batch,
            is_cc=True,
        ).to(self.device0)
        x, adj, rank2, _, diff_traj = self.sampling_fn(
            self.model_x, self.model_adj, self.model_rank2, self.init_flags
        )

        for _ in range(1, self.divide_batch):
            x_, adj_, rank2_, _, _ = self.sampling_fn(
                self.model_x, self.model_adj, self.model_rank2, self.init_flags
            )
            x = torch.cat((x, x_), dim=0)
            adj = torch.cat((adj, adj_), dim=0)
            rank2 = torch.cat((rank2, rank2_), dim=0)

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
        print("Sampling done.")
        sampling_time = perf_counter() - start_sampling_time
        time_log(logger, time_type="sample", elapsed_time=sampling_time)
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            wandb.log({"Sampling time": sampling_time})

        # -------- Save generated molecules --------
        with open(os.path.join(self.log_dir, f"{self.log_name}_smiles.txt"), "a") as f:
            for smiles in gen_smiles:
                f.write(f"{smiles}\n")

        # -------- Evaluation --------
        # Eval molecules
        scores = get_all_metrics(
            gen=gen_smiles,
            k=len(gen_smiles),
            device=self.device_score,
            n_jobs=8,
            test=test_smiles,
            train=train_smiles,
        )
        scores_nspdk = eval_graph_list(
            self.test_graph_list,
            gen_graph_list,
            methods=["nspdk"],
            folder=self.config.folder,
        )["nspdk"]

        # Eval CCs
        methods, kernels = load_cc_eval_settings()
        result_dict_CC = eval_CC_list(
            self.test_CC_list,
            gen_CC_list,
            worker_kwargs=self.worker_kwargs,
            methods=methods,
            kernels=kernels,
            cc_nb_eval=self.cc_nb_eval,
        )

        logger.log(
            f"CCs Eval @{self.cc_nb_eval} {result_dict_CC}", verbose=False
        )  # verbose=False cause already printed
        logger.log(f"Number of molecules: {num_mols}")
        logger.log(f"validity w/o correction: {num_mols_wo_correction / num_mols}")
        for metric in ["valid", f"unique@{len(gen_smiles)}", "FCD/Test", "Novelty"]:
            logger.log(f"{metric}: {scores[metric]}")
        logger.log(f"NSPDK MMD: {scores_nspdk}")
        logger.log(100 * "=")
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add scores to wandb
            wandb.log(
                {
                    "validity": scores["valid"],
                    f"unique@{len(gen_smiles)}": scores[f"unique@{len(gen_smiles)}"],
                    "FCD/Test": scores["FCD/Test"],
                    "Novelty": scores["Novelty"],
                    "NSPDK MMD": scores_nspdk,
                }
            )
            wandb.log(result_dict_CC)

        # -------- Save samples & Plot --------
        # Ccs
        save_dir = save_cc_list(
            self.config, self.log_folder_name, self.log_name + "_mol_ccs", gen_CC_list
        )
        with open(save_dir, "rb") as f:
            sample_CC_list = pickle.load(f)
        plot_cc_list(
            config=self.config,
            ccs=sample_CC_list,
            title=f"mol_ccs_{self.log_name}",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(
                    *[self.config.folder, "samples", "fig", self.log_folder_name]
                ),
                f"mol_ccs_{self.log_name}.png",
            )
            wandb.log({"Generated Mol Combinatorial Complexes": wandb.Image(img_path)})
        # Graphs
        save_dir = save_graph_list(
            self.config,
            self.log_folder_name,
            self.log_name + "_mol_graphs",
            gen_graph_list,
        )
        with open(save_dir, "rb") as f:
            sample_graph_list = pickle.load(f)
        plot_graphs_list(
            config=self.config,
            graphs=sample_graph_list,
            title=f"mol_graphs_{self.log_name}",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(
                    *[self.config.folder, "samples", "fig", self.log_folder_name]
                ),
                f"mol_graphs_{self.log_name}.png",
            )
            wandb.log({"Generated Mol Graphs": wandb.Image(img_path)})
        # Molecules
        save_dir = save_molecule_list(
            self.config, self.log_folder_name, self.log_name + "_mols", gen_mols
        )
        with open(save_dir, "rb") as f:
            sample_mol_list = pickle.load(f)
        plot_molecule_list(
            config=self.config,
            mols=sample_mol_list,
            title=f"mols_{self.log_name}",
            max_num=16,
            save_dir=self.log_folder_name,
        )
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(
                    *[self.config.folder, "samples", "fig", self.log_folder_name]
                ),
                f"mols_{self.log_name}.png",
            )
            wandb.log({"Generated Molecules": wandb.Image(img_path)})
        # 3D Molecule
        if self.config.general_config.plotly_fig:
            molecule = gen_mols[0]
            mol_3d = plot_3D_molecule(molecule)
            filedir = os.path.join(
                *[self.config.folder, "samples", "fig", self.log_folder_name]
            )
            filename = f"mols_3d_{self.log_name}.gif"
            rotate_molecule_animation(
                mol_3d,
                filedir=filedir,
                filename=filename,
                duration=1.0,
                frames=30,
                rotations_per_sec=1.0,
                overwrite=True,
                engine=self.config.general_config.engine,
            )
            if (
                self.config.experiment_type == "train"
            ) and self.config.general_config.use_wandb:
                # add plots to wandb
                img_path = os.path.join(filedir, filename)
                wandb.log({"Generated Molecule 3D": wandb.Image(img_path)})
        # Diffusion trajectory animation - Graphs and molecules
        if self.config.general_config.plotly_fig:
            # Graph
            filedir = os.path.join(
                *[self.config.folder, "samples", "fig", self.log_folder_name]
            )
            filename = f"diff_traj_graphs_{self.log_name}.gif"
            diffusion_animation(
                diff_traj=diff_traj,
                is_molecule=False,
                filedir=filedir,
                filename=filename,
                fps=25,
                overwrite=True,
                engine=self.config.general_config.engine,
            )
            if (
                self.config.experiment_type == "train"
            ) and self.config.general_config.use_wandb:
                # add plots to wandb
                img_path = os.path.join(filedir, filename)
                wandb.log({"Diffusion Trajectory Graph": wandb.Image(img_path)})
            # Graph Cropped
            filename = f"diff_traj_graphs_cropped_{self.log_name}.gif"
            diffusion_animation(
                diff_traj=diff_traj,
                is_molecule=False,
                filedir=filedir,
                filename=filename,
                fps=25,
                overwrite=True,
                engine=self.config.general_config.engine,
                cropped=True,
            )
            if (
                self.config.experiment_type == "train"
            ) and self.config.general_config.use_wandb:
                # add plots to wandb
                img_path = os.path.join(filedir, filename)
                wandb.log({"Diffusion Trajectory Graph Cropped": wandb.Image(img_path)})
            # Molecule
            filename = f"diff_traj_mol_{self.log_name}.gif"
            diffusion_animation(
                diff_traj=diff_traj,
                is_molecule=True,
                filedir=filedir,
                filename=filename,
                fps=25,
                overwrite=True,
                engine=self.config.general_config.engine,
            )
            if (
                self.config.experiment_type == "train"
            ) and self.config.general_config.use_wandb:
                # add plots to wandb
                img_path = os.path.join(filedir, filename)
                wandb.log({"Diffusion Trajectory Molecule": wandb.Image(img_path)})
            # Molecule Cropped
            filename = f"diff_traj_mol_cropped_{self.log_name}.gif"
            diffusion_animation(
                diff_traj=diff_traj,
                is_molecule=True,
                filedir=filedir,
                filename=filename,
                fps=25,
                overwrite=True,
                engine=self.config.general_config.engine,
                cropped=True,
            )
            if (
                self.config.experiment_type == "train"
            ) and self.config.general_config.use_wandb:
                # add plots to wandb
                img_path = os.path.join(filedir, filename)
                wandb.log(
                    {"Diffusion Trajectory Molecule Cropped": wandb.Image(img_path)}
                )


def get_sampler_from_config(
    config: EasyDict,
) -> Sampler:
    """Get the sampler from a configuration file config

    Args:
        config (EasyDict): configuration file

    Returns:
        Sampler: sampler to use for the experiment
    """
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

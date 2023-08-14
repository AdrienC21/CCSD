#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""trainer.py: code for training the model.
"""

import abc
import os
import pickle
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import wandb
from easydict import EasyDict
from tqdm import tqdm, trange

from ccsd.src.utils.loader import (
    load_batch,
    load_data,
    load_device,
    load_ema,
    load_loss_fn,
    load_model_optimizer,
    load_model_params,
    load_seed,
)
from ccsd.src.utils.logger import (
    Logger,
    device_log,
    model_parameters_log,
    set_log,
    start_log,
    train_log,
)
from ccsd.src.utils.plot import plot_lc


class Trainer(abc.ABC):
    """Abstract class for a Trainer."""

    def __init__(self, config: Optional[EasyDict]) -> None:
        """Initialize the trainer.

        Args:
            config (Optional[EasyDict], optional): the config object to use. Defaults to None.
        """
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def train(self, ts: str) -> str:
        """Train method to load the models, the optimizers, etc, train the model and save the checkpoint.

        Args:
            ts (str): checkpoint name (usually a timestamp)

        Returns:
            str: checkpoint name
        """
        pass

    def save_learning_curves(self, learning_curves: Dict[str, List[float]]) -> None:
        """Save the learning curves in a .npy file.

        Args:
            learning_curves (Dict[str, List[float]]): the learning curves to save
        """
        log_name = f"{self.config.config_name}_{self.ckpt}"
        with open(
            os.path.join(self.log_dir, f"{log_name}_learning_curves.npy"), "wb"
        ) as f:
            pickle.dump(learning_curves, f, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_learning_curves(self, learning_curves: Dict[str, List[float]]) -> None:
        """Plot the learning curves.

        Args:
            learning_curves (Dict[str, List[float]]): the learning curves to plot
        """

        # Call the plot function from utils
        log_name = f"{self.config.config_name}_{self.ckpt}"
        plot_lc(
            self.config, learning_curves, self.log_dir, f"{log_name}_learning_curves"
        )


class Trainer_Graph(Trainer):
    """Trainer class for training the model with graphs.

    Adapted from Jo, J. & al (2022)
    """

    def __init__(self, config: EasyDict) -> None:
        """Initialize the trainer with the different configs.

        Args:
            config (EasyDict): the config object to use
        """
        super(Trainer_Graph, self).__init__(config)

        # Load general config
        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(
            self.config, is_train=True, folder=self.config.folder
        )
        self.is_cc = self.config.is_cc

        # Load training config
        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.train_loader, self.test_loader = load_data(self.config)
        self.params_x, self.params_adj = load_model_params(self.config)

    def __repr__(self) -> str:
        """Return the string representation of the Trainer_Graph class.

        Returns:
            str: the string representation of the Trainer_Graph class
        """
        return f"{self.__class__.__name__}(is_cc={self.is_cc})"

    def train(self, ts: str) -> str:
        """Train method to load the models, the optimizers, etc, train the model and save the checkpoint.

        Args:
            ts (str): checkpoint name (usually a timestamp)

        Returns:
            str: checkpoint name
        """
        self.config.exp_name = ts
        self.ckpt = f"{ts}"
        print("\033[91m" + f"{self.ckpt}" + "\033[0m")

        # -------- Load models, optimizers, ema --------
        self.model_x, self.optimizer_x, self.scheduler_x = load_model_optimizer(
            self.params_x, self.config.train, self.device
        )
        self.model_adj, self.optimizer_adj, self.scheduler_adj = load_model_optimizer(
            self.params_adj, self.config.train, self.device
        )
        self.ema_x = load_ema(self.model_x, decay=self.config.train.ema)
        self.ema_adj = load_ema(self.model_adj, decay=self.config.train.ema)

        log_name = f"{self.config.config_name}_{self.ckpt}"
        logger = Logger(str(os.path.join(self.log_dir, f"{log_name}.log")), mode="a")
        logger.log(f"{self.ckpt}", verbose=False)
        device_log(logger, self.device)
        start_log(logger, self.config)
        train_log(logger, self.config)
        model_parameters_log(logger, [self.model_x, self.model_adj])

        self.loss_fn = load_loss_fn(self.config)

        # -------- Training --------
        print("Training started...")
        for epoch in trange(
            0, (self.config.train.num_epochs), desc="[Epoch]", position=1, leave=False
        ):
            self.train_x = []
            self.train_adj = []
            self.test_x = []
            self.test_adj = []
            t_start = time.time()

            self.model_x.train()
            self.model_adj.train()

            for _, train_b in enumerate(self.train_loader):
                self.optimizer_x.zero_grad()
                self.optimizer_adj.zero_grad()
                x, adj = load_batch(train_b, self.device, is_cc=self.is_cc)
                loss_subject = (x, adj)

                loss_x, loss_adj = self.loss_fn(
                    self.model_x, self.model_adj, *loss_subject
                )
                loss_x.backward()
                loss_adj.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model_x.parameters(), self.config.train.grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.model_adj.parameters(), self.config.train.grad_norm
                )

                self.optimizer_x.step()
                self.optimizer_adj.step()

                # -------- EMA update --------
                self.ema_x.update(self.model_x.parameters())
                self.ema_adj.update(self.model_adj.parameters())

                self.train_x.append(loss_x.item())
                self.train_adj.append(loss_adj.item())

            if self.config.train.lr_schedule:
                self.scheduler_x.step()
                self.scheduler_adj.step()

            self.model_x.eval()
            self.model_adj.eval()
            for _, test_b in enumerate(self.test_loader):
                x, adj = load_batch(test_b, self.device, is_cc=self.is_cc)
                loss_subject = (x, adj)

                with torch.no_grad():
                    self.ema_x.store(self.model_x.parameters())
                    self.ema_x.copy_to(self.model_x.parameters())
                    self.ema_adj.store(self.model_adj.parameters())
                    self.ema_adj.copy_to(self.model_adj.parameters())

                    loss_x, loss_adj = self.loss_fn(
                        self.model_x, self.model_adj, *loss_subject
                    )
                    self.test_x.append(loss_x.item())
                    self.test_adj.append(loss_adj.item())

                    self.ema_x.restore(self.model_x.parameters())
                    self.ema_adj.restore(self.model_adj.parameters())

            mean_train_x = np.mean(self.train_x)
            mean_train_adj = np.mean(self.train_adj)
            mean_test_x = np.mean(self.test_x)
            mean_test_adj = np.mean(self.test_adj)

            # -------- Log losses --------
            # Logger
            logger.log(
                f"{epoch+1:03d} | {time.time()-t_start:.2f}s | "
                f"test x: {mean_test_x:.3e} | test adj: {mean_test_adj:.3e} | "
                f"train x: {mean_train_x:.3e} | train adj: {mean_train_adj:.3e} | ",
                verbose=False,
            )

            # Wandb
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "time": time.time() - t_start,
                    "test_x_loss": mean_test_x,
                    "test_adj_loss": mean_test_adj,
                    "train_x_loss": mean_train_x,
                    "train_adj_loss": mean_train_adj,
                }
            )

            # -------- Save checkpoints --------
            if (
                epoch % self.config.train.save_interval
                == self.config.train.save_interval - 1
            ):
                save_name = (
                    f"_{epoch+1}" if epoch < self.config.train.num_epochs - 1 else ""
                )

                torch.save(
                    {
                        "model_config": self.config,
                        "params_x": self.params_x,
                        "params_adj": self.params_adj,
                        "x_state_dict": self.model_x.state_dict(),
                        "adj_state_dict": self.model_adj.state_dict(),
                        "ema_x": self.ema_x.state_dict(),
                        "ema_adj": self.ema_adj.state_dict(),
                    },
                    f"./checkpoints/{self.config.data.data}/{self.ckpt + save_name}.pth",
                )

            if (
                epoch % self.config.train.print_interval
                == self.config.train.print_interval - 1
            ):
                tqdm.write(
                    f"[EPOCH {epoch+1:04d}] test adj: {mean_test_adj:.3e} | train adj: {mean_train_adj:.3e} | "
                    f"test x: {mean_test_x:.3e} | train x: {mean_train_x:.3e}"
                )
        print("Training complete.")
        # -------- Save final model --------
        torch.save(
            {
                "model_config": self.config,
                "params_x": self.params_x,
                "params_adj": self.params_adj,
                "x_state_dict": self.model_x.state_dict(),
                "adj_state_dict": self.model_adj.state_dict(),
                "ema_x": self.ema_x.state_dict(),
                "ema_adj": self.ema_adj.state_dict(),
            },
            f"./checkpoints/{self.config.data.data}/{self.ckpt}_final.pth",
        )
        # -------- Save learning curves and plots --------
        learning_curves = {
            "train_x": self.train_x,
            "train_adj": self.train_adj,
            "test_x": self.test_x,
            "test_adj": self.test_adj,
        }
        self.save_learning_curves(learning_curves)
        self.plot_learning_curves(learning_curves)
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(*[self.log_dir, "fig"]),
                f"{self.config.config_name}_{self.ckpt}_learning_curves.png",
            )
            wandb.log({"Learning Curves": wandb.Image(img_path)})
        return f"{self.ckpt}_final"


class Trainer_CC(Trainer):
    """Trainer class for training the model with combinatorial complexes."""

    def __init__(self, config: EasyDict) -> None:
        """Initialize the trainer with the different configs.

        Args:
            config (EasyDict): the config object to use
        """
        super(Trainer_CC, self).__init__(config)

        # Load general config
        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(
            self.config, is_train=True, folder=self.config.folder
        )
        self.is_cc = self.config.is_cc

        # Load training config
        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.train_loader, self.test_loader = load_data(self.config, is_cc=True)
        self.params_x, self.params_adj, self.params_rank2 = load_model_params(
            self.config, is_cc=True
        )

    def __repr__(self) -> str:
        """Return the string representation of the Trainer_CC class.

        Returns:
            str: the string representation of the Trainer_CC class
        """
        return f"{self.__class__.__name__}(is_cc={self.is_cc})"

    def train(self, ts: str) -> str:
        """Train method to load the models, the optimizers, etc, train the model and save the checkpoint.

        Args:
            ts (str): checkpoint name (usually a timestamp)

        Returns:
            str: checkpoint name
        """
        self.config.exp_name = ts
        self.ckpt = f"{ts}"
        print("\033[91m" + f"{self.ckpt}" + "\033[0m")

        # -------- Load models, optimizers, ema --------
        self.model_x, self.optimizer_x, self.scheduler_x = load_model_optimizer(
            self.params_x, self.config.train, self.device
        )
        self.model_adj, self.optimizer_adj, self.scheduler_adj = load_model_optimizer(
            self.params_adj, self.config.train, self.device
        )
        (
            self.model_rank2,
            self.optimizer_rank2,
            self.scheduler_rank2,
        ) = load_model_optimizer(self.params_rank2, self.config.train, self.device)
        self.ema_x = load_ema(self.model_x, decay=self.config.train.ema)
        self.ema_adj = load_ema(self.model_adj, decay=self.config.train.ema)
        self.ema_rank2 = load_ema(self.model_rank2, decay=self.config.train.ema)

        log_name = f"{self.config.config_name}_{self.ckpt}"
        logger = Logger(str(os.path.join(self.log_dir, f"{log_name}.log")), mode="a")
        logger.log(f"{self.ckpt}", verbose=False)
        device_log(logger, self.device)
        start_log(logger, self.config)
        train_log(logger, self.config)
        model_parameters_log(logger, [self.model_x, self.model_adj, self.model_rank2])

        self.loss_fn = load_loss_fn(self.config, is_cc=True)

        # -------- Training --------
        print("Training started...")
        for epoch in trange(
            0, (self.config.train.num_epochs), desc="[Epoch]", position=1, leave=False
        ):
            self.train_x = []
            self.train_adj = []
            self.train_rank2 = []
            self.test_x = []
            self.test_adj = []
            self.test_rank2 = []
            t_start = time.time()

            self.model_x.train()
            self.model_adj.train()
            self.model_rank2.train()

            for _, train_b in enumerate(self.train_loader):
                self.optimizer_x.zero_grad()
                self.optimizer_adj.zero_grad()
                self.optimizer_rank2.zero_grad()
                x, adj, rank2 = load_batch(train_b, self.device, is_cc=self.is_cc)
                loss_subject = (x, adj, rank2)

                loss_x, loss_adj, loss_rank2 = self.loss_fn(
                    self.model_x, self.model_adj, self.model_rank2, *loss_subject
                )
                loss_x.backward()
                loss_adj.backward()
                loss_rank2.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model_x.parameters(), self.config.train.grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.model_adj.parameters(), self.config.train.grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.model_rank2.parameters(), self.config.train.grad_norm
                )

                self.optimizer_x.step()
                self.optimizer_adj.step()
                self.optimizer_rank2.step()

                # -------- EMA update --------
                self.ema_x.update(self.model_x.parameters())
                self.ema_adj.update(self.model_adj.parameters())
                self.ema_rank2.update(self.model_rank2.parameters())

                self.train_x.append(loss_x.item())
                self.train_adj.append(loss_adj.item())
                self.train_rank2.append(loss_rank2.item())

            if self.config.train.lr_schedule:
                self.scheduler_x.step()
                self.scheduler_adj.step()
                self.scheduler_rank2.step()

            self.model_x.eval()
            self.model_adj.eval()
            self.model_rank2.eval()
            for _, test_b in enumerate(self.test_loader):
                x, adj, rank2 = load_batch(test_b, self.device, is_cc=self.is_cc)
                loss_subject = (x, adj, rank2)

                with torch.no_grad():
                    self.ema_x.store(self.model_x.parameters())
                    self.ema_x.copy_to(self.model_x.parameters())
                    self.ema_adj.store(self.model_adj.parameters())
                    self.ema_adj.copy_to(self.model_adj.parameters())
                    self.ema_rank2.store(self.model_rank2.parameters())
                    self.ema_rank2.copy_to(self.model_rank2.parameters())

                    loss_x, loss_adj, loss_rank2 = self.loss_fn(
                        self.model_x, self.model_adj, self.model_rank2, *loss_subject
                    )
                    self.test_x.append(loss_x.item())
                    self.test_adj.append(loss_adj.item())
                    self.test_rank2.append(loss_rank2.item())

                    self.ema_x.restore(self.model_x.parameters())
                    self.ema_adj.restore(self.model_adj.parameters())
                    self.ema_rank2.restore(self.model_rank2.parameters())

            mean_train_x = np.mean(self.train_x)
            mean_train_adj = np.mean(self.train_adj)
            mean_train_rank2 = np.mean(self.train_rank2)
            mean_test_x = np.mean(self.test_x)
            mean_test_adj = np.mean(self.test_adj)
            mean_test_rank2 = np.mean(self.test_rank2)

            # -------- Log losses --------
            logger.log(
                f"{epoch+1:03d} | {time.time()-t_start:.2f}s | "
                f"test x: {mean_test_x:.3e} | test adj: {mean_test_adj:.3e} | test rank2: {mean_test_rank2:.3e} | "
                f"train x: {mean_train_x:.3e} | train adj: {mean_train_adj:.3e} | train rank2: {mean_train_rank2:.3e} |",
                verbose=False,
            )

            # -------- Save checkpoints --------
            if (
                epoch % self.config.train.save_interval
                == self.config.train.save_interval - 1
            ):
                save_name = (
                    f"_{epoch+1}" if epoch < self.config.train.num_epochs - 1 else ""
                )

                torch.save(
                    {
                        "model_config": self.config,
                        "params_x": self.params_x,
                        "params_adj": self.params_adj,
                        "params_rank2": self.params_rank2,
                        "x_state_dict": self.model_x.state_dict(),
                        "adj_state_dict": self.model_adj.state_dict(),
                        "rank2_state_dict": self.model_rank2.state_dict(),
                        "ema_x": self.ema_x.state_dict(),
                        "ema_adj": self.ema_adj.state_dict(),
                        "ema_rank2": self.ema_rank2.state_dict(),
                    },
                    f"./checkpoints/{self.config.data.data}/{self.ckpt + save_name}.pth",
                )

            if (
                epoch % self.config.train.print_interval
                == self.config.train.print_interval - 1
            ):
                tqdm.write(
                    f"[EPOCH {epoch+1:04d}] test adj: {mean_test_adj:.3e} | train adj: {mean_train_adj:.3e} | "
                    f"test x: {mean_test_x:.3e} | train x: {mean_train_x:.3e} | "
                    f"test rank2: {mean_test_rank2:.3e} | train rank2: {mean_train_rank2:.3e}"
                )
        print("Training complete.")
        # -------- Save final model --------
        torch.save(
            {
                "model_config": self.config,
                "params_x": self.params_x,
                "params_adj": self.params_adj,
                "params_rank2": self.params_rank2,
                "x_state_dict": self.model_x.state_dict(),
                "adj_state_dict": self.model_adj.state_dict(),
                "rank2_state_dict": self.model_rank2.state_dict(),
                "ema_x": self.ema_x.state_dict(),
                "ema_adj": self.ema_adj.state_dict(),
                "ema_rank2": self.ema_rank2.state_dict(),
            },
            f"./checkpoints/{self.config.data.data}/{self.ckpt}_final.pth",
        )
        # -------- Save learning curves and plots --------
        learning_curves = {
            "train_x": self.train_x,
            "train_adj": self.train_adj,
            "train_rank2": self.train_rank2,
            "test_x": self.test_x,
            "test_adj": self.test_adj,
            "test_rank2": self.test_rank2,
        }
        self.save_learning_curves(learning_curves)
        self.plot_learning_curves(learning_curves)
        if (
            self.config.experiment_type == "train"
        ) and self.config.general_config.use_wandb:
            # add plots to wandb
            img_path = os.path.join(
                os.path.join(*[self.log_dir, "fig"]),
                f"{self.config.config_name}_{self.ckpt}_learning_curves.png",
            )
            wandb.log({"Learning Curves": wandb.Image(img_path)})
        return f"{self.ckpt}_final"


def get_trainer_from_config(
    config: EasyDict,
) -> Trainer:
    """Get the trainer from a configuration file config

    Args:
        config (EasyDict): configuration file

    Returns:
        Trainer: trainer to use for the experiment
    """
    if config.is_cc:
        trainer = Trainer_CC(config)
    else:
        trainer = Trainer_Graph(config)
    return trainer

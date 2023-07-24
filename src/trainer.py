#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""trainer.py: code for training the model.
"""

import os
import time
from typing import Union

import numpy as np
import torch
from easydict import EasyDict
from tqdm import tqdm, trange

from src.utils.loader import (
    load_seed,
    load_device,
    load_data,
    load_model_params,
    load_model_optimizer,
    load_ema,
    load_loss_fn,
    load_batch,
)
from src.utils.logger import Logger, set_log, start_log, train_log


class Trainer(object):
    """Trainer class for training the model with graphs."""

    def __init__(self, config: EasyDict) -> None:
        """Initialize the trainer with the different configs.

        Args:
            config (EasyDict): the config object to use
        """
        super(Trainer, self).__init__()

        # Load general config
        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)

        # Load training config
        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.train_loader, self.test_loader = load_data(self.config)
        self.params_x, self.params_adj = load_model_params(self.config)

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

        logger = Logger(str(os.path.join(self.log_dir, f"{self.ckpt}.log")), mode="a")
        logger.log(f"{self.ckpt}", verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)

        self.loss_fn = load_loss_fn(self.config)

        # -------- Training --------
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
                x, adj = load_batch(train_b, self.device)
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
                x, adj = load_batch(test_b, self.device)
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
            logger.log(
                f"{epoch+1:03d} | {time.time()-t_start:.2f}s | "
                f"test x: {mean_test_x:.3e} | test adj: {mean_test_adj:.3e} | "
                f"train x: {mean_train_x:.3e} | train adj: {mean_train_adj:.3e} | ",
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
        print(" ")
        return self.ckpt


class Trainer_CC(object):
    """Trainer class for training the model with combinatorial complexes."""

    def __init__(self, config: EasyDict) -> None:
        """Initialize the trainer with the different configs.

        Args:
            config (EasyDict): the config object to use
        """
        super(Trainer_CC, self).__init__()

        # Load general config
        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)

        # Load training config
        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.train_loader, self.test_loader = load_data(self.config, is_cc=True)
        self.params_x, self.params_adj, self.params_rank2 = load_model_params(
            self.config, is_CC=True
        )

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

        logger = Logger(str(os.path.join(self.log_dir, f"{self.ckpt}.log")), mode="a")
        logger.log(f"{self.ckpt}", verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config)

        self.loss_fn = load_loss_fn(self.config, is_cc=True)

        # -------- Training --------
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
                x, adj, rank2 = load_batch(train_b, self.device)
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
                x, adj, rank2 = load_batch(test_b, self.device, is_cc=True)
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
        print(" ")
        return self.ckpt


def get_trainer_from_config(
    config: EasyDict,
) -> Union[Trainer, Trainer_CC]:
    if config.is_cc:
        trainer = Trainer_CC
    else:
        trainer = Trainer
    return trainer

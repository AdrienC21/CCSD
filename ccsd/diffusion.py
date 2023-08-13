#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ccsd.py: Code for the CCSD class for training and/or sampling.
"""

import os
from time import perf_counter

import wandb
from easydict import EasyDict

from .src.parsers.config import get_config, get_general_config
from .src.sampler import Sampler, get_sampler_from_config
from .src.trainer import Trainer, get_trainer_from_config
from .src.utils.print import initial_print
from .src.utils.time_utils import get_time


class CCSD:
    """CCSD class for training and/or sampling."""

    def __init__(
        self,
        type: str,
        config: str,
        folder: str = "./",
        comment: str = "",
        seed: int = 42,
    ) -> None:
        """Initialize the CCSD class.

        Args:
            type (str): Type of experiment. Choose from ["train", "sample"].
            config (str): Path of config file
            folder (str, optional): Directory to save the results, load checkpoints, load config, etc. Defaults to "./".
            comment (str, optional): A single line comment for the experiment. Defaults to "".
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        # Check the type and config
        assert type in (
            "train",
            "sample",
        ), f"Unknown type: {type}. Please select from [train, sample]."
        if config[-5:] == ".yaml":
            config = config[:-5]
        assert os.path.exists(
            os.path.join(folder, "config", f"{config}.yaml")
        ), f"Config {config} not found."

        # General experiment parameters
        self.type = type
        self.config = config
        self.folder = folder
        self.comment = comment
        self.seed = seed
        self.args = {
            "type": type,
            "config": config,
            "folder": folder,
            "comment": comment,
            "seed": seed,
        }

        # Objects saved during the experiment
        self.cfg = None  # config dictionary
        self.trainer = None  # trainer object
        self.sampler = None  # sampler object

    def __repr__(self) -> str:
        """Representation of the CCSD class.

        Returns:
            str: representation of the CCSD class.
        """
        string_repr = (
            f"{self.__class__.__name__}("
            f"type={self.type}, "
            f"config={self.config}, "
            f"folder={self.folder}, "
            f"comment={self.comment}, "
            f"seed={self.seed}"
        )
        if self.trainer is not None:
            string_repr += f", trainer={self.trainer.__class__.__name__}"
        if self.sampler is not None:
            string_repr += f", sampler={self.sampler.__class__.__name__}"
        string_repr += ")"
        return string_repr

    def run(self) -> None:
        """Run the code for training and/or sampling.

        Raises:
            ValueError: raise and error the experiment type is not one of [train, sample].
        """
        # Get the configuration and the general configuration
        config = get_config(self.args.config, self.args.seed)
        general_config = get_general_config()

        # Print the initial message
        if general_config.print_initial:
            initial_print(self.args)

        # Current timestamp (name of the experiment)
        timezone = general_config.timezone
        ts = get_time(timezone)
        # Add some information to the config
        config.current_time = ts  # add the timestamp to the config
        config.experiment_type = self.args.type  # add the experiment type to the config
        config.config_name = self.args.config  # add the config name to the config
        config.general_config = general_config  # add the general config to the config
        config.folder = self.args.folder  # add the folder to the config

        self.cfg = config  # save the config object

        # -------- Train --------
        if self.args.type == "train":
            start_train_time = perf_counter()
            # Initialize wandb
            if general_config.use_wandb:
                run_name = f"{self.args.config}_{ts}"
                wandb.init(
                    project=general_config.project_name,
                    entity=general_config.entity,
                    config=config,
                    name=run_name,
                )
                wandb.run.name = run_name
                wandb.run.save()
                wandb.config.update(config)
            # Train the model
            # Select the trainer based on the config
            trainer = get_trainer_from_config(config)
            # Train the model
            ckpt = trainer.train(ts)
            print(
                f"Training time: {round(perf_counter() - start_train_time, 3)} seconds"
            )
            self.trainer = trainer  # save the trainer object
            if "sample" in config.keys():  # then sample from the trained model
                start_sampling_time = perf_counter()
                config.ckpt = ckpt  # load the model that has just been trained
                self.cfg = config  # save the updated config object
                # Select the sampler based on the config
                sampler = get_sampler_from_config(config)
                # Sample from the model
                sampler.sample()
                print(
                    f"Sampling time: {round(perf_counter() - start_sampling_time, 3)} seconds"
                )
                self.sampler = sampler  # save the sampler object
            # Finish wandb
            wandb.finish()

        # -------- Generation --------
        elif self.args.type == "sample":
            start_sampling_time = perf_counter()
            # Select the sampler based on the config
            sampler = get_sampler_from_config(config)
            # Sample from the model
            sampler.sample()
            print(
                f"Sampling time: {round(perf_counter() - start_sampling_time, 3)} seconds"
            )
            self.sampler = sampler  # save the sampler object

        else:
            raise ValueError(
                f"Unknown type: {self.args.type}. Please read the documentation and select from [train, sample]."
            )

    def is_trained(self) -> bool:
        """Check if the CCSD model is trained.

        Returns:
            bool: True if the model is trained, False otherwise.
        """
        return self.trainer is not None

    def get_trainer(self) -> Trainer:
        """Get the trainer object.

        Returns:
            Trainer: Trainer object.
        """
        return self.trainer

    def get_sampler(self) -> Sampler:
        """Get the sampler object.

        Returns:
            Sampler: Sampler object.
        """
        return self.sampler

    def get_config(self) -> EasyDict:
        """Get the config object.

        Returns:
            EasyDict: Config object.
        """
        return self.cfg

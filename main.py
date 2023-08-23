#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""main.py: Run the code for training and/or sampling.
Run this script with -h flag to see usage on how to run an experiment.
The arguments are:
    --type: [train, sample] to train the model or sample from a trained model.
    --config: path to the configuration file.

Pipeline structure adapted from Jo, J. & al (2022)
"""

import argparse
import warnings

import matplotlib
import plotly
import wandb
from rdkit import RDLogger

from ccsd.src.parsers.config import get_config, get_general_config
from ccsd.src.parsers.parser import Parser
from ccsd.src.sampler import get_sampler_from_config
from ccsd.src.trainer import get_trainer_from_config
from ccsd.src.utils.print import initial_print
from ccsd.src.utils.time_utils import get_time

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
plotly.io.kaleido.scope.mathjax = None
RDLogger.DisableLog("rdApp.*")


def main(args: argparse.Namespace) -> None:
    """Run the code for training and/or sampling.

    Args:
        args (argparse.Namespace): parsed arguments for the experiment.

    Raises:
        ValueError: raise and error the experiment type is not one of [train, sample].
    """

    # Get the configuration and the general configuration
    config = get_config(args.config, args.seed)
    general_config = get_general_config()

    # Print the initial message
    if general_config.print_initial:
        initial_print(args)

    # Current timestamp (name of the experiment)
    timezone = general_config.timezone
    ts = get_time(timezone)
    # Add some information to the config
    config.current_time = ts  # add the timestamp to the config
    config.experiment_type = args.type  # add the experiment type to the config
    config.config_name = args.config  # add the config name to the config
    config.general_config = general_config  # add the general config to the config
    config.folder = args.folder  # add the folder to the config

    # -------- Train --------
    if args.type == "train":
        # Initialize wandb
        if general_config.use_wandb:
            run_name = f"{args.config}_{ts}"
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
        if "sample" in config.keys():  # then sample from the trained model
            config.ckpt = ckpt  # oad the model that has just been trained
            # Select the sampler based on the config
            sampler = get_sampler_from_config(config)
            # Sample from the model
            sampler.sample()
        # Finish wandb
        wandb.finish()

    # -------- Generation --------
    elif args.type == "sample":
        # Select the sampler based on the config
        sampler = get_sampler_from_config(config)
        # Sample from the model
        sampler.sample()

    else:
        raise ValueError(
            f"Unknown type: {args.type}. Please read the documentation and select from [train, sample]."
        )


if __name__ == "__main__":
    # Parse the input arguments
    args = Parser().parse()
    # Run our training and/or sampling
    main(args)

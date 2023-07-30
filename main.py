#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""main.py: Run the code for training and/or sampling.
Run this script with -h flag to see usage on how to run an experiment.
The arguments are:
    --type: [train, sample] to train the model or sample from a trained model.
    --config: path to the configuration file.
"""

import argparse

import wandb

from src.parsers.config import get_config, get_general_config
from src.parsers.parser import Parser
from src.utils.print import initial_print
from src.utils.time_utils import get_time
from src.sampler import get_sampler_from_config
from src.trainer import get_trainer_from_config


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
    config.config_name = args.config  # add the config name to the config

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
        trainer = get_trainer_from_config(config)
        ckpt = trainer.train(ts)
        # Finish wandb
        wandb.finish()
        if "sample" in config.keys():  # then sample from the trained model
            config.ckpt = ckpt  # to load the model just trained
            sampler = get_sampler_from_config(config)
            sampler.sample()

    # -------- Generation --------
    elif args.type == "sample":
        # Select the sampler based on the dataset
        sampler = get_sampler_from_config(config)
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

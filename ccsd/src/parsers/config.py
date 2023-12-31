#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""config.py: code for loading the config file.

Adapted from Jo, J. & al (2022)
"""

import os

import yaml
from easydict import EasyDict


def get_config(config: str, seed: int, folder: str = "./") -> EasyDict:
    """Load the config file.

    Args:
        config (str): name of the config file.
        seed (int): random seed (to be added to the config object).
        folder (str, optional): folder where the config folder is located. Defaults to "./".

    Returns:
        EasyDict: configuration object.
    """
    config_dir = os.path.join(folder, "config", f"{config}.yaml")
    config = EasyDict(yaml.load(open(config_dir, "r"), Loader=yaml.FullLoader))
    config.seed = seed

    return config


def get_general_config(folder: str = "./") -> EasyDict:
    """Get the general configuration.

    Args:
        folder (str, optional): folder where the config folder is located. Defaults to "./".

    Returns:
        EasyDict: general configuration.
    """
    config_dir = os.path.join(folder, "config", "general_config.yaml")
    config = EasyDict(yaml.load(open(config_dir, "r"), Loader=yaml.FullLoader))

    return config

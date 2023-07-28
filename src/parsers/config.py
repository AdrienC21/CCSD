#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""config.py: code for loading the config file.
"""

import yaml
from easydict import EasyDict


def get_config(config: str, seed: int) -> EasyDict:
    """Load the config file.

    Args:
        config (str): name of the config file.
        seed (int): random seed (to be added to the config object).

    Returns:
        EasyDict: configuration object.
    """
    config_dir = f"./config/{config}.yaml"
    config = EasyDict(yaml.load(open(config_dir, "r"), Loader=yaml.FullLoader))
    config.seed = seed

    return config


def get_general_config() -> EasyDict:
    """Get the general configuration.

    Returns:
        EasyDict: general configuration.
    """
    config_dir = f"./config/general_config.yaml"
    config = EasyDict(yaml.load(open(config_dir, "r"), Loader=yaml.FullLoader))

    return config

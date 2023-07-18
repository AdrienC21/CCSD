#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""config.py: code for loading the config file.
"""

import yaml
from easydict import EasyDict


def get_config(config: str, seed: int) -> EasyDict:
    config_dir = f"./config/{config}.yaml"
    config = EasyDict(yaml.load(open(config_dir, "r"), Loader=yaml.FullLoader))
    config.seed = seed

    return config

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""models_utils.py: utility functions related to the models.
"""

from typing import Union

import torch


def get_model_device(model: Union[torch.nn.Module, torch.nn.DataParallel]) -> str:
    """Get the the device on which the model is loaded ("cpu", "cuda", etc?)

    Args:
        model (Union[torch.nn.Module, torch.nn.DataParallel]): Pytorch model

    Returns:
        str: device on which the model is loaded
    """

    return next(model.parameters()).device.type


def get_nb_parameters(model: torch.nn.Module) -> int:
    """Get the number of parameters of the model.

    Args:
        model (torch.nn.Module): model.

    Returns:
        int: number of parameters of the model.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

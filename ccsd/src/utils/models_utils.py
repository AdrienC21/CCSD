#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""models_utils.py: utility functions related to the models.
"""

from functools import lru_cache
from typing import Sequence, Union

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


@lru_cache(maxsize=64)
def get_ones_cache(shape: Sequence[int], device: str) -> torch.Tensor:
    """Cached function to get a tensor of ones of the given shape and device.

    Args:
        shape (Sequence[int]): shape of the tensor
        device (str): device on which the tensor should be allocated

    Returns:
        torch.Tensor: tensor of ones of the given shape and device
    """
    return torch.ones(shape, dtype=torch.float32, device=device)


def get_ones(shape: Sequence[int], device: str) -> torch.Tensor:
    """Function to get a tensor of ones of the given shape and device.
    Call the cached version of the function and clone it.

    Args:
        shape (Sequence[int]): shape of the tensor
        device (str): device on which the tensor should be allocated

    Returns:
        torch.Tensor: tensor of ones of the given shape and device
    """
    return get_ones_cache(shape, device).clone()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_models_utils.py: test functions for models_utils.py
"""

import torch

from ccsd.src.utils.models_utils import get_model_device, get_nb_parameters, get_ones


def test_get_model_device() -> None:
    """Test the get_model_device function"""
    model = torch.nn.Sequential(torch.nn.Linear(8, 8))  # create a dummy model

    # Test CPU
    model = model.to("cpu")
    assert get_model_device(model) == "cpu"

    # Test GPU (if available)
    if torch.cuda.is_available():
        model = model.to("cuda")
        assert get_model_device(model) == "cuda"


def test_get_nb_parameters() -> None:
    """Test the get_nb_parameters function"""
    model = torch.nn.Sequential(torch.nn.Linear(8, 8))  # create a dummy model
    assert get_nb_parameters(model) == 72


def test_get_ones() -> None:
    """Test the get_ones function"""
    ones = get_ones((10, 3), "cpu")
    assert ones.device.type == "cpu"
    assert ones.shape == (10, 3)
    assert torch.allclose(ones, torch.ones((10, 3), dtype=torch.float32))

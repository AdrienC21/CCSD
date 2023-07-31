#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_models_utils.py: test functions for models_utils.py
"""

import torch

from src.utils.models_utils import get_model_device


def test_get_model_device() -> None:
    """Test the get_model_device function
    """
    model = torch.nn.Sequential(torch.nn.Linear(8, 8))  # create a dummy model
    
    # Test CPU
    model = model.to("cpu")
    assert get_model_device(model) == "cpu"

    # Test GPU (if available)
    if torch.cuda.is_available():
        model = model.to("cuda")
        assert get_model_device(model) == "cuda"

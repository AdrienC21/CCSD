#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_data_loader.py: test functions for data_loader.py
"""

import torch
import pytest

from ccsd.src.utils.ema import (
    ExponentialMovingAverage,
)


@pytest.fixture
def ema_instance() -> ExponentialMovingAverage:
    """Create some initial parameters for testing the EMA class

    Returns:
        ExponentialMovingAverage: _description_
    """
    parameters = [
        torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True),
        torch.nn.Parameter(torch.tensor([2.0]), requires_grad=True),
        torch.nn.Parameter(torch.tensor([3.0]), requires_grad=True),
    ]
    decay = 0.9
    return ExponentialMovingAverage(parameters, decay)


def test_initialization(ema_instance: ExponentialMovingAverage) -> None:
    """Test if the EMA instance is correctly initialized

    Args:
        ema_instance (ExponentialMovingAverage): EMA instance to be tested
    """
    decay = 0.9
    assert ema_instance.decay == decay
    assert ema_instance.num_updates == 0
    assert len(ema_instance.shadow_params) == 3


def test_update(ema_instance: ExponentialMovingAverage) -> None:
    """Test if the EMA instance is correctly updated

    Args:
        ema_instance (ExponentialMovingAverage): EMA instance to be tested
    """
    updated_params = [
        torch.nn.Parameter(torch.tensor([1.5]), requires_grad=True),
        torch.nn.Parameter(torch.tensor([2.5]), requires_grad=True),
        torch.nn.Parameter(torch.tensor([3.5]), requires_grad=True),
    ]
    ema_instance.update(updated_params)

    # Test if the shadow_params are correctly updated
    expected_shadow_params = [
        torch.tensor([1.4091]),  # (1.0 + (1-new_decay) * (1.5 - 1.0))
        torch.tensor([2.4091]),  # (2.0 + (1-new_decay) * (2.5 - 2.0))
        torch.tensor([3.4091]),  # (3.0 + (1-new_decay) * (3.5 - 3.0))
    ]

    for s_param, expected_param in zip(
        ema_instance.shadow_params, expected_shadow_params
    ):
        assert torch.allclose(s_param, expected_param)


def test_copy_to(ema_instance: ExponentialMovingAverage) -> None:
    """Test if the EMA instance correctly copies the shadow_params to the new parameters

    Args:
        ema_instance (ExponentialMovingAverage): EMA instance to be tested
    """
    # Perform one update
    updated_params = [
        torch.nn.Parameter(torch.tensor([1.5]), requires_grad=True),
        torch.nn.Parameter(torch.tensor([2.5]), requires_grad=True),
        torch.nn.Parameter(torch.tensor([3.5]), requires_grad=True),
    ]
    ema_instance.update(updated_params)

    # Create another set of parameters
    new_params = [
        torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True),
        torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True),
        torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True),
    ]
    # Copy EMA's shadow_params to the new_params
    ema_instance.copy_to(new_params)

    # Test if new_params are correctly updated with shadow_params
    expected_new_params = [
        torch.tensor([1.4091]),
        torch.tensor([2.4091]),
        torch.tensor([3.4091]),
    ]

    for param, expected_param in zip(new_params, expected_new_params):
        assert torch.allclose(param, expected_param)


def test_store_and_restore(ema_instance: ExponentialMovingAverage) -> None:
    """Test if the EMA instance correctly stores and restores the parameters

    Args:
        ema_instance (ExponentialMovingAverage): EMA instance to be tested
    """
    # Perform one update
    updated_params = [
        torch.nn.Parameter(torch.tensor([1.5]), requires_grad=True),
        torch.nn.Parameter(torch.tensor([2.5]), requires_grad=True),
        torch.nn.Parameter(torch.tensor([3.5]), requires_grad=True),
    ]
    ema_instance.update(updated_params)

    # Store the current parameters
    ema_instance.store(updated_params)

    # Perform another update
    updated_params = [
        torch.nn.Parameter(torch.tensor([1.7]), requires_grad=True),
        torch.nn.Parameter(torch.tensor([2.7]), requires_grad=True),
        torch.nn.Parameter(torch.tensor([3.7]), requires_grad=True),
    ]
    ema_instance.update(updated_params)

    # Restore the previously stored parameters
    ema_instance.restore(updated_params)

    # Test if the restored parameters are correct
    expected_restored_params = [
        torch.tensor([1.5]),
        torch.tensor([2.5]),
        torch.tensor([3.5]),
    ]

    for param, expected_param in zip(updated_params, expected_restored_params):
        assert torch.allclose(param, expected_param)


def test_state_dict_load_state_dict(ema_instance: ExponentialMovingAverage) -> None:
    """Test if the EMA instance correctly saves and loads the state

    Args:
        ema_instance (ExponentialMovingAverage): EMA instance to be tested
    """
    state_dict = ema_instance.state_dict()

    # Create a new EMA instance with different initial parameters
    new_parameters = [
        torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True),
        torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True),
        torch.nn.Parameter(torch.tensor([0.3]), requires_grad=True),
    ]
    new_ema = ExponentialMovingAverage(new_parameters, ema_instance.decay)
    new_ema.load_state_dict(state_dict)

    # Test if the new EMA instance has the same state as the original EMA
    assert new_ema.decay == ema_instance.decay
    assert new_ema.num_updates == ema_instance.num_updates
    for new_s_param, s_param in zip(new_ema.shadow_params, ema_instance.shadow_params):
        assert torch.allclose(new_s_param, s_param)

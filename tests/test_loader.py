#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_loader.py: test functions for loader.py
"""

import os
import random
from typing import Any, Tuple

import pytest
import torch
import numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader

from src.utils.ema import ExponentialMovingAverage
from src.utils.loader import (
    load_seed,
    load_device,
    load_model,
    load_model_optimizer,
    load_ema,
    load_ema_from_ckpt,
    load_data,
    load_batch,
    load_sde,
    load_loss_fn,
    load_sampling_fn,
    load_model_params,
    load_ckpt,
    load_model_from_ckpt,
    load_eval_settings,
)
from src.utils.models_utils import get_model_device


@pytest.fixture
def mock_torch_cuda_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the torch.cuda.is_available() function to always return True

    Args:
        monkeypatch (pytest.MonkeyPatch): MonkeyPatch object for pytest fixture
    """

    # This fixture will mock the torch.cuda.is_available() function
    # to make it always return True for testing purposes.
    def mock_is_available():
        return True

    monkeypatch.setattr(torch.cuda, "is_available", mock_is_available)


def test_load_seed() -> None:
    """Test the load_seed function"""
    seed = 42
    result = load_seed(seed)

    # Check that the random seeds have been set correctly
    assert random.random() == 0.6394267984578837
    assert np.random.get_state()[1][0] == seed
    assert torch.initial_seed() == seed

    if torch.cuda.is_available():
        assert torch.cuda.initial_seed() == seed
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    # Check that the returned value is the same as the input seed
    assert result == seed


def test_load_device(mock_torch_cuda_is_available: None) -> None:
    """Test the load_device function

    Args:
        mock_torch_cuda_is_available (None): mock the torch.cuda.is_available() function
    """
    device = load_device()

    if torch.cuda.is_available():
        # If CUDA is available, the device should be a list of GPU indices.
        assert isinstance(device, list)
        assert len(device) == torch.cuda.device_count()
    else:
        # If CUDA is not available, the device should be "cpu".
        assert device == "cpu"


def test_load_model() -> None:
    """Test the load_model function"""
    # Test with valid model_type
    params = {
        "model_type": "ScoreNetworkX",
        "max_feat_num": 10,
        "depth": 20,
        "nhid": 5,
    }
    model = load_model(params)
    assert isinstance(model, torch.nn.Module)

    # Test with invalid model_type
    params = {
        "model_type": "InvalidModelType",
        "param1": 10,
        "param2": 20,
    }
    with pytest.raises(ValueError):
        load_model(params)


class MockModel(torch.nn.Module):
    """Mock model for testing purposes"""

    def __init__(self) -> None:
        """Initialize the mock model"""
        super(MockModel, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the mock model

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.fc(x)


@pytest.fixture
def mock_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the load_model function to return a simple mock model

    Args:
        monkeypatch (pytest.MonkeyPatch): MonkeyPatch object for pytest fixture
    """

    # This fixture will mock the load_model function to return a simple mock model
    def mock_load_model_func(params: Any) -> torch.nn.Module:
        """Mock the load_model function to return a simple mock model

        Args:
            params (Any): parameters for the model

        Returns:
            torch.nn.Module: mock model
        """
        return MockModel()

    monkeypatch.setattr("src.utils.loader.load_model", mock_load_model_func)


def test_load_model_optimizer(mock_load_model):
    """Test the load_model_optimizer function

    Args:
        mock_load_model (Callable): mock the load_model function
    """
    params = {
        "model_type": "ScoreNetworkX",
        "max_feat_num": 10,
        "depth": 20,
        "nhid": 5,
    }
    config_train = EasyDict(
        {
            "name": "test",
            "num_epochs": 300,
            "save_interval": 100,
            "print_interval": 1000,
            "reduce_mean": False,
            "lr": 0.005,
            "lr_schedule": True,
            "ema": 0.999,
            "weight_decay": 0.0001,
            "grad_norm": 1.0,
            "lr_decay": 0.999,
            "eps": 1.0e-5,
        }
    )
    device = "cuda:0"

    model, optimizer, scheduler = load_model_optimizer(params, config_train, device)

    # Check if the returned objects are of the correct types
    assert isinstance(model, torch.nn.Module)
    assert isinstance(optimizer, torch.optim.Optimizer)

    if scheduler is not None:
        assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)


def test_load_ema() -> None:
    """Test the load_ema function"""
    model = MockModel()
    ema = load_ema(model)

    # Check if the returned object is of the correct type
    assert isinstance(ema, ExponentialMovingAverage)


def test_load_ema_from_ckpt() -> None:
    """Test the load_ema_from_ckpt function"""
    model = MockModel()
    # Parameters for the EMA at a given checkpoint
    ema_state_dict = {"decay": 0.9, "num_updates": 0, "shadow_params": {}}
    ema = load_ema_from_ckpt(model, ema_state_dict, decay=0.8)

    # Check if the returned object is of the correct type
    assert isinstance(ema, ExponentialMovingAverage)


def test_load_data() -> None:
    """Test the load_data function"""
    # Assuming we have valid configurations for the datasets
    config_qm9 = EasyDict(
        {"is_cc": False, "data": {"dir": "data", "data": "QM9", "batch_size": 32}}
    )

    # Test with a dataset that is not a CombinatorialComplex
    data_loader = load_data(config_qm9)
    assert isinstance(data_loader[0], DataLoader)
    assert isinstance(data_loader[1], DataLoader)


@pytest.fixture
def create_mock_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    """Mock a simple batch tensor for testing purposes

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: a tuple of tensors (x_b, adj_b)
    """
    x_b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    adj_b = torch.tensor([[0, 1], [1, 0]])
    return x_b, adj_b


@pytest.fixture
def mock_load_sde(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the load_sde function to return a simple SDE object

    Args:
        monkeypatch (pytest.MonkeyPatch): MonkeyPatch object for pytest fixture
    """

    def mock_load_sde_func(config_sde: EasyDict) -> Any:
        class MockSDE:
            """Mock SDE class for testing purposes"""

            def __init__(self) -> None:
                """Initialize the mock SDE class"""
                pass

        return MockSDE()

    monkeypatch.setattr("src.utils.loader.load_sde", mock_load_sde_func)


def test_load_batch(create_mock_batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
    """Test the load_batch function

    Args:
        create_mock_batch (Tuple[torch.Tensor, torch.Tensor]): a tuple of tensors (x_b, adj_b)
    """
    device = "cpu"  # Assume we are using CPU for testing
    x_b, adj_b = create_mock_batch
    batch = [x_b, adj_b]

    # Test without is_cc=True
    x_loaded, adj_loaded = load_batch(batch, device)
    assert torch.equal(x_loaded, x_b.to(device))
    assert torch.equal(adj_loaded, adj_b.to(device))

    # Test with is_cc=True
    rank2_b = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    batch_cc = [x_b, adj_b, rank2_b]
    x_loaded, adj_loaded, rank2_loaded = load_batch(batch_cc, device, is_cc=True)
    assert torch.equal(x_loaded, x_b.to(device))
    assert torch.equal(adj_loaded, adj_b.to(device))
    assert torch.equal(rank2_loaded, rank2_b.to(device))


def test_load_sde(mock_load_sde: Any) -> None:
    """Test the load_sde function

    Args:
        mock_load_sde (Any): mock the load_sde function
    """
    config_sde = torch.nn.ParameterDict(
        {"type": "VP", "beta_min": 0.1, "beta_max": 1.0, "num_scales": 3}
    )
    sde = load_sde(config_sde)
    assert hasattr(
        sde, "__init__"
    )  # Check if the returned object is a class (simple SDE mock).


def test_load_loss_fn(mock_load_sde: Any) -> None:
    """Test the load_loss_fn function

    Args:
        mock_load_sde (Any): mock the load_loss_fn function
    """
    config = EasyDict(
        {"train": {"reduce_mean": True, "eps": 1e-6}, "sde": {"x": {}, "adj": {}}}
    )
    loss_fn = load_loss_fn(config)

    # Check if the returned object is a callable function.
    assert callable(loss_fn)

    # Test with is_cc=True
    config_cc = EasyDict(config.copy())
    config_cc.sde.rank2 = {}
    config_cc.data = {}
    config_cc.data.d_min = 3
    config_cc.data.d_max = 9
    loss_fn_cc = load_loss_fn(config_cc, is_cc=True)
    assert callable(loss_fn_cc)


@pytest.fixture
def create_mock_configs() -> Tuple[EasyDict, EasyDict, EasyDict, EasyDict, str]:
    """Create mock configurations for testing purposes

    Returns:
        Tuple[EasyDict, EasyDict, EasyDict, EasyDict, str]: a tuple of mock configurations
    """
    config_train = EasyDict(
        {
            "is_cc": False,
            "data": {
                "data": "QM9",
                "dir": "./data",
                "batch_size": 1024,
                "max_node_num": 9,
                "max_feat_num": 4,
                "init": "atom",
            },
            "sde": {
                "x": {
                    "type": "VE",
                    "beta_min": 0.1,
                    "beta_max": 1.0,
                    "num_scales": 1000,
                },
                "adj": {
                    "type": "VE",
                    "beta_min": 0.1,
                    "beta_max": 1.0,
                    "num_scales": 1000,
                },
            },
            "model": {
                "x": "ScoreNetworkX",
                "adj": "ScoreNetworkA",
                "conv": "GCN",
                "num_heads": 4,
                "depth": 2,
                "adim": 16,
                "nhid": 16,
                "num_layers": 3,
                "num_linears": 3,
                "c_init": 2,
                "c_hid": 8,
                "c_final": 4,
                "use_bn": False,
            },
            "train": {
                "name": "test",
                "num_epochs": 300,
                "save_interval": 100,
                "print_interval": 1000,
                "reduce_mean": False,
                "lr": 0.005,
                "lr_schedule": True,
                "ema": 0.999,
                "weight_decay": 0.0001,
                "grad_norm": 1.0,
                "lr_decay": 0.999,
                "eps": 1.0e-5,
            },
        }
    )

    config_train_cc = EasyDict(config_train.copy())
    config_train_cc.sde.rank2 = {
        "type": "VE",
        "beta_min": 0.1,
        "beta_max": 1.0,
        "num_scales": 1000,
    }
    config_train_cc.is_cc = True
    config_train_cc.model.rank2 = "ScoreNetworkF"
    config_train_cc.model.num_layers_mlp = 2
    config_train_cc.model.cnum = 3
    config_train_cc.model.use_hodge_mask = True
    config_train_cc.data.d_min = 3
    config_train_cc.data.d_max = 4

    config_module = EasyDict(
        {
            "predictor": "Euler",
            "corrector": "Langevin",
            "snr": 0.1,
            "scale_eps": 1e-6,
            "n_steps": 10,
        }
    )
    config_sample = EasyDict(
        {"probability_flow": False, "noise_removal": False, "eps": 1e-6}
    )
    device = "cpu"  # Assume we are using CPU for testing
    return config_train, config_train_cc, config_module, config_sample, device


def test_load_sampling_fn(
    create_mock_configs: Tuple[EasyDict, EasyDict, EasyDict, EasyDict, str]
) -> None:
    """Test the load_sampling_fn function

    Args:
        create_mock_configs (Tuple[EasyDict, EasyDict, EasyDict, EasyDict, str]): a tuple of mock configurations
    """
    # Assuming we have valid configurations for the function
    (
        config_train,
        config_train_cc,
        config_module,
        config_sample,
        device,
    ) = create_mock_configs

    # Test without is_cc=True
    sampling_fn = load_sampling_fn(config_train, config_module, config_sample, device)
    assert callable(sampling_fn)

    # Test with is_cc=True
    sampling_fn_cc = load_sampling_fn(
        config_train_cc,
        config_module,
        config_sample,
        device,
        is_cc=True,
        d_min=config_train_cc.data.d_min,
        d_max=config_train_cc.data.d_max,
    )
    assert callable(sampling_fn_cc)


def test_load_model_params(
    create_mock_configs: Tuple[EasyDict, EasyDict, EasyDict, EasyDict, str]
) -> None:
    """Test the load_model_params function"""
    # Assuming we have valid configurations for the function
    config, config_cc, _, _, _ = create_mock_configs

    # Test without is_cc=True
    params_x, params_adj = load_model_params(config)
    assert isinstance(params_x, dict)
    assert isinstance(params_adj, dict)

    # Test with is_cc=True
    params_x, params_adj, params_rank2 = load_model_params(config_cc, is_cc=True)
    assert isinstance(params_x, dict)
    assert isinstance(params_adj, dict)
    assert isinstance(params_rank2, dict)


@pytest.fixture
def create_mock_config() -> Tuple[EasyDict, str, str]:
    """Create a mock configuration for testing purposes

    Returns:
        Tuple[EasyDict, str, str]: a tuple of mock configuration, device, and timestamp
    """
    config = EasyDict(
        {"ckpt": "test_data", "data": {"data": "QM9"}, "sample": {"use_ema": True}}
    )
    device = "cpu"  # Assume we are using CPU for testing
    ts = "test_data"
    return config, device, ts


def test_load_ckpt(create_mock_config: Tuple[EasyDict, str, str]) -> None:
    """Test the load_ckpt function

    Args:
        create_mock_config (Tuple[EasyDict, str, str]): a tuple of mock configurations
    """
    # Assuming we have a valid configuration for the function
    config, device, ts = create_mock_config

    # Create a mock checkpoint file
    path = f"./checkpoints/{config.data.data}/{config.ckpt}.pth"
    ckpt = {
        "model_config": {"model_type": "ScoreNetworkX", "nhid": 10, "depth": 20},
        "params_x": {},
        "x_state_dict": {},
        "params_adj": {},
        "adj_state_dict": {},
        "params_rank2": {},
        "rank2_state_dict": {},
        "ema_x": {},
        "ema_adj": {},
        "ema_rank2": {},
    }
    with open(path, "wb") as f:
        torch.save(ckpt, f)

    # Test without is_cc=True
    ckpt_dict = load_ckpt(config, device, ts, return_ckpt=True)
    assert isinstance(ckpt_dict, dict)

    # Test with is_cc=True
    ckpt_dict_cc = load_ckpt(config, device, ts, return_ckpt=True, is_cc=True)
    assert isinstance(ckpt_dict_cc, dict)

    # Clean up the checkpoint file
    os.remove(path)


def test_load_model_from_ckpt() -> None:
    """Test the load_model_from_ckpt function"""
    # Assuming we have valid parameters and state_dict for the function
    params = {"model_type": "ScoreNetworkX", "max_feat_num": 1, "nhid": 1, "depth": 1}
    state_dict = EasyDict(
        {
            "module.layers.0.weight": torch.tensor([[0.7769]]),
            "module.layers.0.bias": torch.tensor([0.0]),
            "module.final.linears.0.weight": torch.tensor(
                [
                    [0.0131, -0.2723],
                    [0.5632, -0.1321],
                    [-0.1254, 0.2206],
                    [-0.6828, 0.6731],
                ]
            ),
            "module.final.linears.0.bias": torch.tensor(
                [0.3538, 0.0530, -0.3859, 0.5845]
            ),
            "module.final.linears.1.weight": torch.tensor(
                [
                    [-0.3185, 0.3213, 0.0781, -0.0056],
                    [-0.1986, -0.1701, 0.1657, -0.1041],
                    [-0.1483, -0.3809, 0.2485, 0.0118],
                    [0.0509, 0.1199, 0.3830, -0.3968],
                ]
            ),
            "module.final.linears.1.bias": torch.tensor(
                [-0.0851, -0.4657, -0.0670, -0.3860]
            ),
            "module.final.linears.2.weight": torch.tensor(
                [[-0.4637, -0.1629, -0.1027, -0.3141]]
            ),
            "module.final.linears.2.bias": torch.tensor([-0.2526]),
        }
    )
    device = "cpu"

    # Test without DataParallel
    model = load_model_from_ckpt(params, state_dict, device)
    assert isinstance(model, torch.nn.Module)
    assert (
        (
            model.state_dict()["final.linears.2.weight"]
            == torch.tensor([[-0.4637, -0.1629, -0.1027, -0.3141]])
        )
        .all()
        .item()
    )

    # Test with cuda if available. If multiple devices, test DataParallel
    if torch.cuda.is_available():
        device = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        model = load_model_from_ckpt(params, state_dict, device)
        if len(device) > 1:
            assert isinstance(model, torch.nn.DataParallel)
        assert get_model_device(model) == "cuda"


def test_load_eval_settings() -> None:
    """Test the load_eval_settings function"""
    # Assuming we have a valid data object for the function
    data = np.random.rand(10, 10)

    # Test with default orbit_on=True
    methods, kernels = load_eval_settings(data)
    assert isinstance(methods, list)
    assert isinstance(kernels, dict)

    # Test with orbit_on=False
    methods, kernels = load_eval_settings(data, orbit_on=False)
    assert isinstance(methods, list)
    assert isinstance(kernels, dict)

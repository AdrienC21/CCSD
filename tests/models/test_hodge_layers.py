#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_hodge_layers.py: test functions for hodge_layers.py
"""

from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from ccsd.src.models.hodge_layers import (
    BaselineBlock,
    DenseHCNConv,
    HodgeBaselineLayer,
    HodgeNetworkLayer,
)
from ccsd.src.utils.cc_utils import (
    adj_to_hodgedual,
    create_incidence_1_2,
    pow_tensor_cc,
)
from ccsd.src.utils.graph_utils import pow_tensor

# Initialize random seeds
torch.manual_seed(42)
np.random.seed(42)


@pytest.fixture
def create_matrices() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a random input tensor, adjacency matrix, and rank2 incidence matrix for testing.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: node feature matrix, adjacency matrix, and rank2 incidence matrix
    """
    d_min = 3
    d_max = 4
    X = np.array([[np.random.random() for _ in range(10)] for _ in range(5)])
    X = torch.tensor(X, dtype=torch.float32)
    N = X.shape[0]  # 5
    A = torch.zeros((N, N), dtype=torch.float32)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 0)]:
        A[i, j] = 1.0
        A[j, i] = 1.0
    two_rank_cells = {frozenset((0, 1, 2, 3)): {}, frozenset((0, 3, 4)): {}}
    F = create_incidence_1_2(N, A, d_min, d_max, two_rank_cells)
    F = torch.tensor(F, dtype=torch.float32)

    # Add batch dimension
    x = X.unsqueeze(0)
    adj = A.unsqueeze(0)
    rank2 = F.unsqueeze(0)
    return x, adj, rank2


@pytest.fixture
def create_layer_param_config_DenseHCNConv(
    create_matrices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> Dict[str, Any]:
    """Create a layer parameter configuration for testing.
    This is a light one to reduce the value of the output tensor and the testing time.

    Args:
        create_matrices (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): node feature matrix, adjacency matrix, and rank2 incidence matrix

    Returns:
        Dict[str, Any]: layer parameter configuration
    """
    _, _, rank2 = create_matrices
    params = {
        "in_channels": rank2.shape[-1],
        "out_channels": 3,
        "bias": True,
    }
    return params


@pytest.fixture
def create_layer_param_config_HodgeNetworkLayer() -> Dict[str, Any]:
    """Create a layer parameter configuration for testing.
    This is a light one to reduce the value of the output tensor and the testing time.

    Returns:
        Dict[str, Any]: layer parameter configuration
    """
    params = {
        "num_linears": 2,
        "input_dim": 3,
        "nhid": 4,
        "output_dim": 2,
        "d_min": 3,
        "d_max": 4,
        "use_bn": False,
    }
    return params


@pytest.fixture
def create_layer_param_config_BaselineBlock(
    create_matrices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> Dict[str, Any]:
    """Create a layer parameter configuration for testing.
    This is a light one to reduce the value of the output tensor and the testing time.

    Args:
        create_matrices (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): node feature matrix, adjacency matrix, and rank2 incidence matrix

    Returns:
        Dict[str, Any]: layer parameter configuration
    """
    _, _, rank2 = create_matrices
    params = {
        "in_dim": rank2.shape[-2],
        "hidden_dim": 4,
        "out_dim": rank2.shape[-2],
    }
    return params


@pytest.fixture
def create_layer_param_config_HodgeBaselineLayer() -> Dict[str, Any]:
    """Create a layer parameter configuration for testing.
    This is a light one to reduce the value of the output tensor and the testing time.

    Returns:
        Dict[str, Any]: layer parameter configuration
    """
    params = {
        "num_linears": 2,
        "input_dim": 3,
        "hidden_dim": 4,
        "conv_output_dim": 2,
        "N": 5,
        "d_min": 3,
        "d_max": 4,
        "use_bn": False,
    }
    return params


def test_DenseHCNConv(
    create_matrices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    create_layer_param_config_DenseHCNConv: Dict[str, Any],
) -> None:
    """Test the DenseHCNConv class.

    Args:
        create_matrices (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): node feature matrix, adjacency matrix, and rank2 incidence matrix
        create_layer_param_config_DenseHCNConv (Dict[str, Any]): layer parameter configuration
    """
    # Initialize random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Check the model/layer output shape(s) and value(s)
    x, adj, rank2 = create_matrices
    hodge_adj = adj_to_hodgedual(adj)
    params = create_layer_param_config_DenseHCNConv
    layer = DenseHCNConv(**params)
    layer.eval()
    with torch.no_grad():
        out = layer(hodge_adj, rank2, mask=None)

    assert out.shape == (
        hodge_adj.shape[0],
        hodge_adj.shape[-1],
        params["out_channels"],
    )
    expected_out = torch.tensor(
        [
            [
                [0.5214, -0.4904, 0.4457],
                [0.0000, 0.0000, 0.0000],
                [0.4399, -0.0454, 0.5311],
                [-0.0815, 0.4451, 0.0853],
                [0.5214, -0.4904, 0.4457],
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000],
                [0.5214, -0.4904, 0.4457],
                [0.0000, 0.0000, 0.0000],
                [-0.0815, 0.4451, 0.0853],
            ]
        ]
    )
    assert torch.allclose(out, expected_out, atol=1e-4)


def test_HodgeNetworkLayer(
    create_matrices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    create_layer_param_config_HodgeNetworkLayer: Dict[str, Any],
) -> None:
    """Test the HodgeNetworkLayer class.

    Args:
        create_matrices (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): node feature matrix, adjacency matrix, and rank2 incidence matrix
        create_layer_param_config_HodgeNetworkLayer (Dict[str, Any]): layer parameter configuration
    """
    # Initialize random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Check the model/layer output shape(s) and value(s)
    x, adj, rank2 = create_matrices
    params = create_layer_param_config_HodgeNetworkLayer
    rank2c = pow_tensor_cc(rank2, params["input_dim"], None)
    layer = HodgeNetworkLayer(**params)
    layer.eval()
    with torch.no_grad():
        out = layer(
            rank2c,
            adj.shape[-1],
            flags=None,
        )
    assert out.shape == (
        rank2c.shape[0],
        params["output_dim"],
        rank2c.shape[2],
        rank2c.shape[3],
    )
    expected_out_0_0_0 = torch.tensor(
        [
            -0.3947,
            -0.3947,
            -0.3947,
            -0.3947,
            -0.3947,
            -0.2819,
            -0.3947,
            -0.3947,
            -0.3947,
            -0.3947,
            -0.0359,
            -0.3947,
            -0.3947,
            -0.3947,
            -0.3947,
        ]
    )
    assert torch.allclose(out[0, 0, 0], expected_out_0_0_0, atol=1e-4)


def test_BaselineBlock(
    create_matrices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    create_layer_param_config_BaselineBlock: Dict[str, Any],
) -> None:
    """Test the BaselineBlock class.

    Args:
        create_matrices (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): node feature matrix, adjacency matrix, and rank2 incidence matrix
        create_layer_param_config_BaselineBlock (Dict[str, Any]): layer parameter configuration
    """
    # Initialize random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Check the model/layer output shape(s) and value(s)
    x, adj, rank2 = create_matrices
    hodge_adj = adj_to_hodgedual(adj)
    params = create_layer_param_config_BaselineBlock
    layer = BaselineBlock(**params)
    layer.eval()
    with torch.no_grad():
        out_rank2, out_hodge_adj = layer(
            hodge_adj, rank2, flags=None, attention_mask=None
        )
    batch_size, nb_edges, K = rank2.shape
    assert out_rank2.shape == (batch_size, nb_edges, K)
    assert out_hodge_adj.shape == (batch_size, nb_edges, params["out_dim"])

    expected_rank2_0 = torch.tensor(
        [
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.6283,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.1764,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
        ]
    )
    expected_hodge_adj_0 = torch.tensor(
        [
            0.4429,
            -0.0743,
            0.3022,
            -0.0531,
            0.0591,
            0.2948,
            0.3786,
            -0.1217,
            0.3985,
            0.3078,
        ]
    )
    assert torch.allclose(out_rank2[0, 0, :], expected_rank2_0, atol=1e-4)
    assert torch.allclose(out_hodge_adj[0, 0, :], expected_hodge_adj_0, atol=1e-4)


def test_HodgeBaselineLayer(
    create_matrices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    create_layer_param_config_HodgeBaselineLayer: Dict[str, Any],
) -> None:
    """Test the HodgeBaselineLayer class.

    Args:
        create_matrices (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): node feature matrix, adjacency matrix, and rank2 incidence matrix
        create_layer_param_config_HodgeBaselineLayer (Dict[str, Any]): layer parameter configuration
    """
    # Initialize random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Check the model/layer output shape(s) and value(s)
    x, adj, rank2 = create_matrices
    params = create_layer_param_config_HodgeBaselineLayer
    adj = pow_tensor(adj, params["input_dim"])
    hodge_adj = adj_to_hodgedual(adj)
    layer = HodgeBaselineLayer(**params)
    layer.eval()
    with torch.no_grad():
        out_hodge_adj, out_rank2 = layer(hodge_adj, rank2, flags=None)
    assert out_hodge_adj.shape == (
        hodge_adj.shape[0],
        params["conv_output_dim"],
        hodge_adj.shape[-1],
        hodge_adj.shape[-1],
    )
    assert out_rank2.shape == rank2.shape
    expected_out_hodge_adj_0_0_0 = torch.tensor(
        [
            0.2970,
            -0.4434,
            -0.0274,
            0.0460,
            -0.4553,
            0.0315,
            -0.0247,
            -0.3034,
            0.2659,
            -0.1461,
        ]
    )
    expected_out_rank2_0_0 = torch.tensor(
        [
            -0.1114,
            -0.1114,
            -0.1114,
            -0.1114,
            -0.1114,
            -0.1190,
            -0.1114,
            -0.1114,
            -0.1114,
            -0.1114,
            -0.2877,
            -0.1114,
            -0.1114,
            -0.1114,
            -0.1114,
        ]
    )
    assert torch.allclose(
        out_hodge_adj[0, 0, 0], expected_out_hodge_adj_0_0_0, atol=1e-4
    )
    assert torch.allclose(out_rank2[0, 0], expected_out_rank2_0_0, atol=1e-4)

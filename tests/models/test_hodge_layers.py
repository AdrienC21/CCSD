#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_hodge_layers.py: test functions for hodge_layers.py
"""

from typing import Tuple, Dict, Any

import pytest
import torch
import numpy as np

from ccsd.src.utils.cc_utils import (
    create_incidence_1_2,
    adj_to_hodgedual,
    pow_tensor_cc,
)
from ccsd.src.models.hodge_layers import HodgeNetworkLayer, DenseHCNConv


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

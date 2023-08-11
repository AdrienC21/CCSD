#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_hodge_attention.py: test functions for hodge_attention.py
"""

from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from ccsd.src.models.hodge_attention import HodgeAdjAttentionLayer, HodgeAttention
from ccsd.src.utils.cc_utils import adj_to_hodgedual, create_incidence_1_2
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
def create_layer_param_config_HodgeAttention(
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
        "in_dim": rank2.shape[-1],
        "attn_dim": 4,
        "out_dim": 2,
        "num_heads": 4,
        "conv": "HCN",
    }
    return params


@pytest.fixture
def create_layer_param_config_HodgeAdjAttentionLayer() -> Dict[str, Any]:
    """Create a layer parameter configuration for testing.
    This is a light one to reduce the value of the output tensor and the testing time.

    Returns:
        Dict[str, Any]: layer parameter configuration
    """
    params = {
        "num_linears": 2,
        "input_dim": 3,
        "attn_dim": 4,
        "conv_output_dim": 2,
        "N": 5,
        "d_min": 3,
        "d_max": 4,
        "num_heads": 4,
        "conv": "HCN",
        "use_bn": False,
    }
    return params


def test_HodgeAttention(
    create_matrices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    create_layer_param_config_HodgeAttention: Dict[str, Any],
) -> None:
    """Test the HodgeAttention class.

    Args:
        create_matrices (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): node feature matrix, adjacency matrix, and rank2 incidence matrix
        create_layer_param_config_HodgeAttention (Dict[str, Any]): layer parameter configuration
    """
    # Initialize random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Check the model/layer output shape(s) and value(s)
    x, adj, rank2 = create_matrices
    hodge_adj = adj_to_hodgedual(adj)
    params = create_layer_param_config_HodgeAttention
    layer = HodgeAttention(**params)
    layer.eval()
    with torch.no_grad():
        out_value, out_attention = layer(
            hodge_adj, rank2, flags=None, attention_mask=None
        )
    batch_size, nb_edges, K = rank2.shape
    assert out_value.shape == (batch_size, nb_edges, params["out_dim"])
    assert out_attention.shape == (batch_size, nb_edges, nb_edges)

    expected_value_0 = torch.tensor(
        [
            -0.0677,
            0.0000,
            -0.2025,
            -0.1348,
            -0.0677,
            0.0000,
            0.0000,
            -0.0677,
            0.0000,
            -0.1348,
        ]
    )
    expected_attention_0 = torch.tensor(
        [
            -0.0141,
            0.0000,
            0.0186,
            0.0326,
            -0.0141,
            0.0000,
            0.0000,
            -0.0141,
            0.0000,
            0.0326,
        ]
    )
    assert torch.allclose(out_value[0, :, 0], expected_value_0, atol=1e-4)
    assert torch.allclose(out_attention[0, 0, :], expected_attention_0, atol=1e-4)


def test_HodgeAdjAttentionLayer(
    create_matrices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    create_layer_param_config_HodgeAdjAttentionLayer: Dict[str, Any],
) -> None:
    """Test the HodgeAdjAttentionLayer class.

    Args:
        create_matrices (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): node feature matrix, adjacency matrix, and rank2 incidence matrix
        create_layer_param_config_HodgeAdjAttentionLayer (Dict[str, Any]): layer parameter configuration
    """
    # Initialize random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Check the model/layer output shape(s) and value(s)
    x, adj, rank2 = create_matrices
    params = create_layer_param_config_HodgeAdjAttentionLayer
    adj = pow_tensor(adj, params["input_dim"])
    hodge_adj = adj_to_hodgedual(adj)
    layer = HodgeAdjAttentionLayer(**params)
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
            0.5701,
            0.5655,
            0.5712,
            0.5666,
            0.5701,
            0.5655,
            0.5655,
            0.5701,
            0.5655,
            0.5666,
        ]
    )
    expected_out_rank2_0_0 = torch.tensor(
        [
            0.2798,
            0.3881,
            0.3431,
            0.3085,
            0.2685,
            0.2759,
            0.4419,
            0.4073,
            0.3661,
            0.3170,
            0.4945,
            0.3513,
            0.3112,
            0.2753,
            0.3210,
        ]
    )
    assert torch.allclose(
        out_hodge_adj[0, 0, 0], expected_out_hodge_adj_0_0_0, atol=1e-4
    )
    assert torch.allclose(out_rank2[0, 0], expected_out_rank2_0_0, atol=1e-4)

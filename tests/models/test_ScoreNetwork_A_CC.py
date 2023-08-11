#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_ScoreNetwork_A_CC.py: test functions for ScoreNetwork_A_CC.py
"""

from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from ccsd.src.models.ScoreNetwork_A_CC import ScoreNetworkA_CC
from ccsd.src.utils.cc_utils import create_incidence_1_2

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
def create_model_param_config() -> Dict[str, Any]:
    """Create a model parameter configuration for testing.

    Returns:
        Dict[str, Any]: model parameter configuration
    """
    params = {
        "max_feat_num": 10,
        "max_node_num": 5,
        "d_min": 3,
        "d_max": 4,
        "nhid": 4,
        "num_layers": 8,
        "num_linears": 16,
        "c_init": 32,
        "c_hid": 128,
        "c_final": 64,
        "adim": 4,
        "num_heads": 4,
        "conv": "GCN",
        "conv_hodge": "HCN",
        "use_bn": False,
        "is_cc": True,
    }
    return params


@pytest.fixture
def create_model_param_config_v2() -> Dict[str, Any]:
    """Create a model parameter configuration for testing.
    This is a lighter one to reduce the value of the output tensor and the testing time.

    Returns:
        Dict[str, Any]: model parameter configuration
    """
    params = {
        "max_feat_num": 10,
        "max_node_num": 5,
        "d_min": 3,
        "d_max": 4,
        "nhid": 4,
        "num_layers": 2,
        "num_linears": 2,
        "c_init": 2,
        "c_hid": 2,
        "c_final": 2,
        "adim": 2,
        "num_heads": 2,
        "conv": "GCN",
        "conv_hodge": "HCN",
        "use_bn": False,
        "is_cc": True,
    }
    return params


def test_ScoreNetworkA_CC(
    create_matrices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    create_model_param_config: Dict[str, Any],
    create_model_param_config_v2: Dict[str, Any],
) -> None:
    """Test the ScoreNetworkA_CC class.

    Args:
        create_matrices (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): node feature matrix, adjacency matrix, and rank2 incidence matrix
        create_model_param_config (Dict[str, Any]): model parameter configuration
        create_model_param_config_v2 (Dict[str, Any]): model parameter configuration
    """
    # Initialize random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Check the model/layer output shape(s) and value(s)
    x, adj, rank2 = create_matrices
    params = create_model_param_config
    params2 = create_model_param_config_v2
    # Test shape using the first configuration
    model = ScoreNetworkA_CC(**params)
    model.eval()
    with torch.no_grad():
        out = model(x, adj, rank2)
    assert out.shape == (1, 5, 5)

    # Test value using the second configuration
    model = ScoreNetworkA_CC(**params2)
    model.eval()
    with torch.no_grad():
        out = model(x, adj, rank2)
    expected_out = torch.tensor(
        [
            [
                [-0.0000, 0.0071, -0.0212, -0.0339, -0.0357],
                [0.0071, -0.0000, 0.0119, -0.0226, 0.0219],
                [-0.0212, 0.0119, -0.0000, 0.0167, 0.0295],
                [-0.0339, -0.0226, 0.0167, 0.0000, -0.0308],
                [-0.0357, 0.0219, 0.0295, -0.0308, -0.0000],
            ]
        ]
    )
    assert torch.allclose(out, expected_out, atol=1e-4)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_ScoreNetwork_F.py: test functions for ScoreNetwork_F.py
"""

from typing import Tuple, Dict, Any

import pytest
import torch
import numpy as np

from src.utils.cc_utils import create_incidence_1_2
from src.models.ScoreNetwork_F import ScoreNetworkF


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
        "num_layers_mlp": 2,
        "num_layers": 2,
        "num_linears": 2,
        "nhid": 2,
        "c_hid": 3,
        "c_final": 2,
        "cnum": 2,
        "max_node_num": 5,
        "d_min": 3,
        "d_max": 4,
        "use_hodge_mask": True,
        "use_bn": False,
        "is_cc": True,
    }
    return params


def test_ScoreNetworkF(
    create_matrices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    create_model_param_config: Dict[str, Any],
) -> None:
    """Test the ScoreNetworkF class.

    Args:
        create_matrices (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): node feature matrix, adjacency matrix, and rank2 incidence matrix
        create_model_param_config (Dict[str, Any]): model parameter configuration
    """
    # Initialize random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Check the model/layer output shape(s) and value(s)
    x, adj, rank2 = create_matrices
    params = create_model_param_config
    model = ScoreNetworkF(**params)
    model.eval()
    with torch.no_grad():
        out = model(x, adj, rank2, flags=None)

    assert out.shape == rank2.shape
    expected_out_0 = torch.tensor(
        [
            -0.1157,
            -0.1157,
            -0.1157,
            -0.1157,
            -0.1157,
            -0.1157,
            -0.1157,
            -0.1157,
            -0.1157,
            -0.1157,
        ]
    )
    assert torch.allclose(out[0, :, 0], expected_out_0, atol=1e-4)

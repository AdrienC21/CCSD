#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ScoreNetwork_F.py: ScoreNetworkF class.
This is a ScoreNetwork model that operates on the rank2 incidence matrix of the combinatorial complex.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as torch_func

from src.models.layers import DenseGCNConv, MLP
from src.utils.cc_utils import mask_rank2, pow_tensor, mask_x, node_feature_to_matrix
from src.models.attention import AttentionLayer


# TODO: MODIFY THIS CLASS TO IMPLEMENT OUR OWN MODEL


class BaselineNetworkLayer(torch.nn.Module):
    """BaselineNetworkLayer that operates on tensors derived from a rank2 incidence matrix.
    Used in the BaselineNetwork model.
    """

    def __init__(
        self,
    ) -> None:
        """Initialize the BaselineNetworkLayer.
        """
        super(BaselineNetworkLayer, self).__init__()

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, rank2: torch.Tensor, flags: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the BaselineNetworkLayer.

        Args:
            x (torch.Tensor): node feature matrix
            adj (torch.Tensor): adjacency matrix
            rank2 (torch.Tensor): rank2 incidence matrix matrix
            flags (Optional[torch.Tensor]): optional flags for the node features

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: node feature matrix, adjacency matrix, rank2 incidence matrix
        """

        return x, adj, rank2


class BaselineNetwork(torch.nn.Module):
    """BaselineNetwork to calculate the score with respect to the rank2 incidence matrix."""

    def __init__(
        self,
    ) -> None:
        """Initialize the BaselineNetwork.
        """
        super(BaselineNetwork, self).__init__()

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, rank2: torch.Tensor, flags: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the BaselineNetwork. Returns the score with respect to the rank2 incidence matrix.

        Args:
            x (torch.Tensor): node feature matrix
            adj (torch.Tensor): adjacency matrix
            rank2 (torch.Tensor): rank2 incidence matrix
            flags (Optional[torch.Tensor], optional): optional flags for the score. Defaults to None.

        Returns:
            torch.Tensor: score with respect to the rank2 incidence matrix
        """
        score = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        return score


class ScoreNetworkF(BaselineNetwork):
    """ScoreNetworkF to calculate the score with respect to the rank2 incidence matrix."""

    def __init__(
        self,
    ) -> None:
        """Initialize the ScoreNetworkF model.
        """

        super(ScoreNetworkF, self).__init__(
        )

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, rank2: torch.Tensor, flags: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the ScoreNetworkF. Returns the score with respect to the rank2 incidence matrix.

        Args:
            x (torch.Tensor): node feature matrix
            adj (torch.Tensor): adjacency matrix
            rank2 (torch.Tensor): rank2 incidence matrix
            flags (Optional[torch.Tensor], optional): optional flags for the score. Defaults to None.

        Returns:
            torch.Tensor: score with respect to the rank2 incidence matrix
        """
        score = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        return score

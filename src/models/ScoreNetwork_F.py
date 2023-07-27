#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ScoreNetwork_F.py: ScoreNetworkF class.
This is a ScoreNetwork model that operates on the rank2 incidence matrix of the combinatorial complex.
"""

from typing import Optional

import torch
import torch.nn.functional as torch_func

from src.models.layers import MLP
from src.utils.cc_utils import get_rank2_dim, mask_rank2, hodge_laplacian


class ScoreNetworkF(torch.nn.Module):
    """ScoreNetworkF to calculate the score with respect to the rank2 incidence matrix."""

    def __init__(
        self,
    ) -> None:
        """Initialize the ScoreNetworkF model."""
        super(ScoreNetworkF, self).__init__()

        # Initialize the parameters
        self.max_node_num = max_node_num
        self.d_min = d_min
        self.d_max = d_max
        self.rows, self.cols = get_rank2_dim(self.max_node_num, self.d_min, self.d_max)

        # Initialize the layers
        self.layers = torch.nn.ModuleList()

        # Initialize the final MLP
        self.final = MLP(
            num_layers=3,
            input_dim=self.fdim,
            hidden_dim=2 * self.fdim,
            output_dim=1,
            use_bn=False,
            activate_func=torch_func.elu,
        )

        # Initialize the mask
        self.mask = torch.ones((self.rows, self.cols), dtype=torch.float32)
        self.mask.unsqueeze_(0)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        rank2: torch.Tensor,
        flags: Optional[torch.Tensor] = None,
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

        # Apply all the layers
        mat = [hodge_laplacian(rank2)]

        # Apply the final MLP on the concatenated adjacency tensor to compute the score
        batch_size = x.shape[0]
        out_shape = (batch_size, self.rows, self.cols)  # B x (NC2) x K
        score = self.final(mat).view(*out_shape)

        # Mask the score
        self.mask = self.mask.to(score.device)
        score = score * self.mask

        # Mask the score with respect to the flags
        score = mask_rank2(score, self.max_node_num, self.d_min, self.d_max, flags)

        return score

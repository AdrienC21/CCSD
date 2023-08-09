#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""attention.py: DenseHCNConv and HodgeNetworkLayer classes for the ScoreNetwork models and other layers.
"""

from typing import Tuple, Optional

import torch
import torch.nn.functional as torch_func
from torch.nn import Parameter

from src.models.layers import MLP, glorot, zeros
from src.utils.cc_utils import mask_rank2


class HodgeNetworkLayer(torch.nn.Module):
    """HodgeNetworkLayer that operates on tensors derived from a rank2 incidence matrix F.
    Used in the ScoreNetworkF model.
    """

    def __init__(
        self,
        num_linears: int,
        input_dim: int,
        nhid: int,
        output_dim: int,
        d_min: int,
        d_max: int,
        use_bn: bool = False,
    ) -> None:
        """Initialize the HodgeNetworkLayer.

        Args:
            num_linears (int): number of linear layers in the MLP (except the first one)
            input_dim (int): input dimension of the MLP
            nhid (int): number of hidden units in the MLP
            output_dim (int): output dimension of the MLP
            d_min (int): minimum size of the rank2 cells
            d_max (int): maximum size of the rank2 cells
            use_bn (bool, optional): whether to use batch normalization in the MLP. Defaults to False.
        """
        super(HodgeNetworkLayer, self).__init__()

        # Initialize the parameters and the layer(s)
        self.num_linears = num_linears
        self.input_dim = input_dim
        self.nhid = nhid
        self.output_dim = output_dim
        self.d_min = d_min
        self.d_max = d_max
        self.use_bn = use_bn
        self.layer = MLP(
            num_layers=self.num_linears,
            input_dim=self.input_dim,
            hidden_dim=self.nhid,
            output_dim=self.output_dim,
            use_bn=self.use_bn,
            activate_func=torch_func.elu,
        )

        # Initialize the parameters (glorot for the weight and zeros for the bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the HodgeNetworkLayer."""
        # Reset the parameters of the MLP layer
        self.layer.reset_parameters()

    def forward(
        self,
        rank2: torch.Tensor,
        N: int,
        flags: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the HodgeNetworkLayer.

        Args:
            rank2 (torch.Tensor): rank2 incidence matrix
            N (int): maximum number of nodes
            flags (Optional[torch.Tensor]): optional flags for the rank2 incidence matrix

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: node feature matrix, adjacency matrix, and rank2 incidence matrix
        """
        permut_rank2 = rank2.permute(0, 2, 3, 1)
        rank2_out = self.layer(permut_rank2).permute(0, 3, 1, 2)

        # Mask the rank2_out
        rank2_out = mask_rank2(rank2_out, N, self.d_min, self.d_max, flags)

        return rank2_out

    def __repr__(self) -> str:
        """Return a string representation of the HodgeNetworkLayer.

        Returns:
            str: string representation of the HodgeNetworkLayer
        """
        return (
            "{}(layers={}, dim=({}, {}, {}), d_min={}, d_max={}, batch_norm={})".format(
                self.__class__.__name__,
                self.num_linears,
                self.input_dim,
                self.nhid,
                self.output_dim,
                self.d_min,
                self.d_max,
                self.use_bn,
            )
        )


class DenseHCNConv(torch.nn.Module):
    """DenseHCN layer (Hodge Convolutional Network layer)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ) -> None:
        """Initialize the DenseHCNConv layer.

        Args:
            in_channels (int): input channels: must be the the last dimension of a rank-2 incidence matrix
            out_channels (int): output channels: output dimension of the layer, could be an attention dimension or the output dimension of our value matrix (last dimension of a rank-2 incidence matrix)
            bias (bool, optional): if True, add bias parameters. Defaults to True.
        """
        super(DenseHCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))

        # Initialize the bias
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize the parameters (glorot for the weight and zeros for the bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the DenseHCNConv layer.
        Initialize them with Glorot uniform initialization for the weight and zeros for the bias.
        """
        glorot(self.weight)
        zeros(self.bias)

    def __repr__(self) -> str:
        """Return a string representation of the DenseHCNConv layer.

        Returns:
            str: string representation of the DenseHCNConv layer
        """
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )

    def forward(
        self,
        hodge_adj: torch.Tensor,
        rank2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the DenseHCNConv layer.

        Args:
            hodge_adj (torch.Tensor): hodge adjacency matrix (B * (NC2) * (NC2))
            rank2 (torch.Tensor): adjacency matrix (B * (NC2) * K)
            mask (Optional[torch.Tensor], optional): Optional mask for the output. Defaults to None.

        Returns:
            torch.Tensor: output of the DenseHCNConv layer (B * (NC2) * F_o)
        """
        hodge_adj = (
            hodge_adj.unsqueeze(0) if hodge_adj.dim() == 2 else hodge_adj
        )  # batch
        rank2 = rank2.unsqueeze(0) if rank2.dim() == 2 else rank2  # batch
        B = rank2.shape[0]

        out = torch.matmul(rank2, self.weight)
        deg_inv_sqrt = hodge_adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        hodge_adj = deg_inv_sqrt.unsqueeze(-1) * hodge_adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(hodge_adj, out)

        # Add the bias
        if self.bias is not None:
            out = out + self.bias

        # Apply the mask
        if mask is not None:
            out = out * mask.view(B, hodge_adj.shape[1], 1).to(hodge_adj.dtype)

        return out

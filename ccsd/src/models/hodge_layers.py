#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""attention.py: DenseHCNConv and HodgeNetworkLayer classes for the ScoreNetwork models and other layers.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as torch_func
from torch.nn import Parameter

from ccsd.src.models.layers import MLP, glorot, zeros
from ccsd.src.utils.cc_utils import get_rank2_dim, mask_hodge_adjs, mask_rank2


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


class BaselineBlock(torch.nn.Module):
    """Combinatorial Complexes BaselineBlock layer

    Used in the HodgeBaselineLayer below
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        """Initialize the BaselineBlock layer

        Args:
            in_dim (int): input dimension
            hidden_dim (int): hidden dimension
            out_dim (int): output dimension
        """
        super(BaselineBlock, self).__init__()
        # Intialize the parameters of the BaselineBlock layer
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.activation = torch.tanh

        # Define the layer
        self.mlp_layer = MLP(
            num_layers=2,
            input_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.out_dim,
            use_bn=False,
            activate_func=torch_func.elu,
        )

        # Reset the parameters
        self.reset_parameters()

    def __repr__(self) -> str:
        """Representation of the BaselineBlock layer

        Returns:
            str: representation of the BaselineBlock layer
        """
        return (
            f"{self.__class__.__name__}("
            f"in_dim={self.in_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"out_dim={self.out_dim})"
        )

    def reset_parameters(self) -> None:
        """Reset the parameters of the BaselineBlock layer"""
        self.mlp_layer.reset_parameters()

    def forward(
        self,
        hodge_adj: torch.Tensor,
        rank2: torch.Tensor,
        flags: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the BaselineBlock layer. Returns the value, attention matrix.

        Args:
            hodge_adj (torch.Tensor): hodge adjacency matrix
            rank2 (torch.Tensor): rank-2 incidence matrix
            flags (torch.Tensor): node flags
            attention_mask (Optional[torch.Tensor], optional): UNUSED HERE. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: rank2, hodge_adj matrix
        """
        _hodge_adj = self.activation(self.mlp_layer(hodge_adj))
        rank2_out = torch.bmm(_hodge_adj, rank2)

        hodge_adj_out = (
            _hodge_adj + _hodge_adj.transpose(-1, -2)
        ) / 2  # symmetrize the hodge adjacency matrix

        return rank2_out, hodge_adj_out


class HodgeBaselineLayer(torch.nn.Module):
    """HodgeBaselineLayer for ScoreNetworkA_Base_CC with baseline blocks"""

    def __init__(
        self,
        num_linears: int,
        input_dim: int,
        hidden_dim: int,
        conv_output_dim: int,
        N: int,
        d_min: int,
        d_max: int,
        use_bn: bool = False,
    ) -> None:
        """Initialize the HodgeBaselineLayer

        Args:
            num_linears (int): number of linear layers in the MLPs
            input_dim (int): input dimension of the HodgeBaselineLayer (also number of BaselineBlock)
            hidden_dim (int): hidden dimension
            conv_output_dim (int): output dimension of the MLP (output number of channels)
            N (int): maximum number of nodes
            d_min (int): minimum size of rank-2 cells
            d_max (int): maximum size of rank-2 cells
            use_bn (bool, optional): whether to use batch normalization in the MLP. Defaults to False.
        """

        super(HodgeBaselineLayer, self).__init__()
        # Define the parameters of the layer
        self.num_linears = num_linears
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_output_dim = conv_output_dim
        self.N = N
        self.d_min = d_min
        self.d_max = d_max
        self.nb_edges, self.K = get_rank2_dim(N, d_min, d_max)  # calculate nb_edges, K
        self.use_bn = use_bn

        # Define the layers
        self.layers = torch.nn.ModuleList()
        for _ in range(self.input_dim):
            self.layers.append(
                BaselineBlock(
                    self.nb_edges,
                    self.hidden_dim,
                    self.nb_edges,
                )
            )

        self.hidden_dim_mlp = 2 * max(self.input_dim, self.conv_output_dim)
        self.mlp_rank2 = MLP(
            num_layers=self.num_linears,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim_mlp,
            output_dim=1,
            use_bn=self.use_bn,
            activate_func=torch_func.elu,
        )
        self.mlp_hodge = MLP(
            num_layers=self.num_linears,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim_mlp,
            output_dim=self.conv_output_dim,
            use_bn=self.use_bn,
            activate_func=torch_func.elu,
        )

        # Reset the parameters
        self.reset_parameters()

    def __repr__(self) -> str:
        """Representation of the HodgeBaselineLayer

        Returns:
            str: representation of the HodgeBaselineLayer
        """
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"conv_output_dim={self.conv_output_dim}, "
            f"hidden_dim_mlp={self.hidden_dim_mlp})"
        )

    def reset_parameters(self) -> None:
        """Reset the parameters of the HodgeBaselineLayer"""
        # Reset the MLPs
        self.mlp_rank2.reset_parameters()
        self.mlp_hodge.reset_parameters()
        # Reset the attention layers
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
        hodge_adj: torch.Tensor,
        rank2: torch.Tensor,
        flags: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the HodgeBaselineLayer. Returns a hodge adjacency matrix and a rank-2 incidence matrix.

        Args:
            hodge_adj (torch.Tensor): hodge adjacency matrix (B x C_i x (NC2) x (NC2))
                C_i is the number of input channels
            rank2 (torch.Tensor): rank-2 incidence matrix (B x (NC2) x K)
            flags (Optional[torch.Tensor]): flags for the nodes

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: hodge adjacency matrix and a rank-2 incidence matrix (B x C_o x (NC2) x (NC2)), (B x (NC2) x K)
                C_o is the number of output channels
        """

        rank2_list = []
        hodge_list = []
        for k in range(self.input_dim):
            _rank2, _hodge = self.layers[k](hodge_adj[:, k, :, :], rank2, flags)
            rank2_list.append(_rank2.unsqueeze(-1))
            hodge_list.append(_hodge.unsqueeze(-1))
        hodge_adj_out = mask_hodge_adjs(
            self.mlp_hodge(torch.cat(hodge_list, dim=-1)).permute(0, 3, 1, 2),
            flags,
        )
        hodge_adj_out = torch.tanh(hodge_adj_out)
        hodge_adj_out = hodge_adj_out + hodge_adj_out.transpose(-1, -2)

        _rank2_final = self.mlp_rank2(torch.cat(rank2_list, dim=-1)).squeeze(-1)
        rank2_out = mask_rank2(_rank2_final, self.N, self.d_min, self.d_max, flags)

        return hodge_adj_out, rank2_out

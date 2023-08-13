#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""attention.py: HodgeAttention and HodgeAdjAttentionLayer classes for the ScoreNetwork models.
"""

import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as torch_func

from ..utils.cc_utils import get_rank2_dim, mask_hodge_adjs, mask_rank2
from .hodge_layers import DenseHCNConv
from .layers import MLP


class HodgeAttention(torch.nn.Module):
    """Hodge Combinatorial Complexes Multi-Head (HCCMH) Attention layer

    Used in the HodgeAdjAttentionLayer below
    """

    def __init__(
        self,
        in_dim: int,
        attn_dim: int,
        out_dim: int,
        num_heads: int = 4,
        conv: str = "HCN",
    ) -> None:
        """Initialize the HodgeAttention layer

        Args:
            in_dim (int): input dimension
            attn_dim (int): attention dimension
            out_dim (int): output dimension
            num_heads (int, optional): number of attention heads. Defaults to 4.
            conv (str, optional): type of convolutional layer, choose from [HCN, MLP].
                Defaults to "HCN".
        """
        super(HodgeAttention, self).__init__()
        # Intialize the parameters of the HodgeAttention layer
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.attn_dim = attn_dim
        self.out_dim = out_dim
        self.conv = conv
        self.softmax_dim = 2

        # Initialize the GNNs
        self.ccnn_q, self.ccnn_k, self.ccnn_v = self.get_ccnn(
            self.in_dim, self.attn_dim, self.out_dim, self.conv
        )
        self.activation = torch.tanh

        # Reset the parameters of the GNNs
        self.reset_parameters()

    def __repr__(self) -> str:
        """Representation of the HodgeAttention layer

        Returns:
            str: representation of the HodgeAttention layer
        """
        return (
            f"{self.__class__.__name__}("
            f"in_dim={self.in_dim}, "
            f"attn_dim={self.attn_dim}, "
            f"out_dim={self.out_dim}, "
            f"num_heads={self.num_heads}, "
            f"conv={self.conv})"
        )

    def reset_parameters(self) -> None:
        """Reset the parameters of the HodgeAttention layer"""
        self.ccnn_q.reset_parameters()
        self.ccnn_k.reset_parameters()
        self.ccnn_v.reset_parameters()

    def forward(
        self,
        hodge_adj: torch.Tensor,
        rank2: torch.Tensor,
        flags: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the HodgeAttention layer. Returns the value, attention matrix.

        Args:
            hodge_adj (torch.Tensor): hodge adjacency matrix
            rank2 (torch.Tensor): rank-2 incidence matrix
            flags (torch.Tensor): node flags
            attention_mask (Optional[torch.Tensor], optional): attention mask for the attention matrix. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: value, attention matrix
        """

        if self.conv == "HCN":
            Q = self.ccnn_q(hodge_adj, rank2)
            K = self.ccnn_k(hodge_adj, rank2)
        else:
            Q = self.ccnn_q(hodge_adj)
            K = self.ccnn_k(hodge_adj)

        V = self.ccnn_v(hodge_adj, rank2)
        dim_split = self.attn_dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)

        if attention_mask is not None:
            # duplicate the attention mask for each head
            attention_mask = torch.cat(
                [attention_mask for _ in range(self.num_heads)], 0
            )
            # compute the attention score
            attention_score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.out_dim)
            # mask the attention score
            A = self.activation(attention_mask + attention_score)
        else:
            A = self.activation(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.out_dim))

        # A: (B x num_heads) x (NC2) x (NC2)
        A = A.view(-1, *hodge_adj.shape)
        A = A.mean(dim=0)
        A = (A + A.transpose(-1, -2)) / 2  # symmetrize the attention matrix

        return V, A

    def get_ccnn(
        self, in_dim: int, attn_dim: int, out_dim: int, conv: str = "HCN"
    ) -> Tuple[
        Union[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
        Union[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], torch.Tensor],
        ],
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        """Initialize the three CCNNs

        Args:
            in_dim (int): input dimension
            attn_dim (int): attention dimension
            out_dim (int): output dimension
            conv (str, optional): type of convolutional layer, choose from [HCN, MLP].
                Defaults to "HCN".

        Raises:
            NotImplementedError: raise an error if the convolutional layer is not implemented

        Returns:
            Tuple[Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]], Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]: three GNNs, one for the query, one for the key, and one for the value
        """

        if conv == "HCN":
            ccnn_q = DenseHCNConv(in_dim, attn_dim)
            ccnn_k = DenseHCNConv(in_dim, attn_dim)
            ccnn_v = DenseHCNConv(in_dim, out_dim)

            return ccnn_q, ccnn_k, ccnn_v

        elif conv == "MLP":
            num_layers = 2
            ccnn_q = MLP(
                num_layers, in_dim, 2 * attn_dim, attn_dim, activate_func=torch.tanh
            )
            ccnn_k = MLP(
                num_layers, in_dim, 2 * attn_dim, attn_dim, activate_func=torch.tanh
            )
            ccnn_v = DenseHCNConv(in_dim, out_dim)

            return ccnn_q, ccnn_k, ccnn_v

        else:
            raise NotImplementedError(f"Convolution layer {conv} not implemented.")


class HodgeAdjAttentionLayer(torch.nn.Module):
    """HodgeAdjAttentionLayer for ScoreNetworkA_CC"""

    def __init__(
        self,
        num_linears: int,
        input_dim: int,
        attn_dim: int,
        conv_output_dim: int,
        N: int,
        d_min: int,
        d_max: int,
        num_heads: int = 4,
        conv: str = "GCN",
        use_bn: bool = False,
    ) -> None:
        """Initialize the HodgeAdjAttentionLayer

        Args:
            num_linears (int): number of linear layers in the MLPs
            input_dim (int): input dimension of the HodgeAdjAttentionLayer (also number of HodgeAttention)
            attn_dim (int): attention dimension
            conv_output_dim (int): output dimension of the MLP (output number of channels)
            N (int): maximum number of nodes
            d_min (int): minimum size of rank-2 cells
            d_max (int): maximum size of rank-2 cells
            num_heads (int, optional): number of heads for the Attention. Defaults to 4.
            conv (str, optional): type of convolutional layer, choose from [GCN, MLP].
                Defaults to "GCN".
            use_bn (bool, optional): whether to use batch normalization in the MLP. Defaults to False.
        """

        super(HodgeAdjAttentionLayer, self).__init__()
        # Define the parameters of the layer
        self.num_linears = num_linears
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        self.conv_output_dim = conv_output_dim
        self.N = N
        self.d_min = d_min
        self.d_max = d_max
        self.K = get_rank2_dim(N, d_min, d_max)[1]  # calculate K
        self.num_heads = num_heads
        self.conv = conv
        self.use_bn = use_bn

        # Define the layers
        self.attn = torch.nn.ModuleList()
        for _ in range(self.input_dim):
            self.attn.append(
                HodgeAttention(
                    self.K,
                    self.attn_dim,
                    self.K,
                    num_heads=self.num_heads,
                    conv=self.conv,
                )
            )

        self.hidden_dim = 2 * max(self.input_dim, self.conv_output_dim)
        self.mlp_value = MLP(
            num_layers=self.num_linears,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            use_bn=self.use_bn,
            activate_func=torch_func.elu,
        )
        self.mlp_attention = MLP(
            num_layers=self.num_linears,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.conv_output_dim,
            use_bn=self.use_bn,
            activate_func=torch_func.elu,
        )

        # Reset the parameters
        self.reset_parameters()

    def __repr__(self) -> str:
        """Representation of the HodgeAdjAttentionLayer

        Returns:
            str: representation of the HodgeAdjAttentionLayer
        """
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"attn_dim={self.attn_dim}, "
            f"conv_output_dim={self.conv_output_dim}, "
            f"num_heads={self.num_heads}, "
            f"conv={self.conv}, "
            f"hidden_dim={self.hidden_dim})"
        )

    def reset_parameters(self) -> None:
        """Reset the parameters of the HodgeAdjAttentionLayer"""
        # Reset the MLPs
        self.mlp_value.reset_parameters()
        self.mlp_attention.reset_parameters()
        # Reset the attention layers
        for attn in self.attn:
            attn.reset_parameters()

    def forward(
        self,
        hodge_adj: torch.Tensor,
        rank2: torch.Tensor,
        flags: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the HodgeAdjAttentionLayer. Returns a hodge adjacency matrix and a rank-2 incidence matrix.

        Args:
            hodge_adj (torch.Tensor): hodge adjacency matrix (B x C_i x (NC2) x (NC2))
                C_i is the number of input channels
            rank2 (torch.Tensor): rank-2 incidence matrix (B x (NC2) x K)
            flags (Optional[torch.Tensor]): flags for the nodes

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: hodge adjacency matrix and a rank-2 incidence matrix (B x C_o x (NC2) x (NC2)), (B x (NC2) x K)
                C_o is the number of output channels
        """

        value_list = []
        attention_list = []
        for k in range(self.input_dim):
            _value, _attention = self.attn[k](hodge_adj[:, k, :, :], rank2, flags)
            value_list.append(_value.unsqueeze(-1))
            attention_list.append(_attention.unsqueeze(-1))
        hodge_adj_out = mask_hodge_adjs(
            self.mlp_attention(torch.cat(attention_list, dim=-1)).permute(0, 3, 1, 2),
            flags,
        )
        hodge_adj_out = torch.tanh(hodge_adj_out)
        hodge_adj_out = hodge_adj_out + hodge_adj_out.transpose(-1, -2)

        _rank2 = self.mlp_value(torch.cat(value_list, dim=-1)).squeeze(-1)
        rank2_out = mask_rank2(_rank2, self.N, self.d_min, self.d_max, flags)

        return hodge_adj_out, rank2_out

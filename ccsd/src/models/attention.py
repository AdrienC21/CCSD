#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""attention.py: Attention and AttentionLayer classes for the ScoreNetwork models.

Adapted from Jo, J. & al (2022)

Almost left untouched.
"""

import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ccsd.src.models.layers import MLP, DenseGCNConv
from ccsd.src.utils.graph_utils import mask_adjs, mask_x


class Attention(torch.nn.Module):
    """Graph Multi-Head (GMH) Attention layer

    Adapted from Baek et al. (2021)

    Used in the AttentionLayer below
    """

    def __init__(
        self,
        in_dim: int,
        attn_dim: int,
        out_dim: int,
        num_heads: int = 4,
        conv: str = "GCN",
    ) -> None:
        """Initialize the Attention layer

        Args:
            in_dim (int): input dimension
            attn_dim (int): attention dimension
            out_dim (int): output dimension
            num_heads (int, optional): number of attention heads. Defaults to 4.
            conv (str, optional): type of convolutional layer, choose from [GCN, MLP].
                Defaults to "GCN".
        """
        super(Attention, self).__init__()
        # Intialize the parameters of the Attention layer
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.out_dim = out_dim
        self.conv = conv
        self.softmax_dim = 2

        # Initialize the GNNs
        self.gnn_q, self.gnn_k, self.gnn_v = self.get_gnn(
            in_dim, attn_dim, out_dim, conv
        )
        self.activation = torch.tanh

        # Reset the parameters of the GNNs
        self.reset_parameters()

    def __repr__(self) -> str:
        """Representation of the Attention layer

        Returns:
            str: representation of the Attention layer
        """
        return (
            f"{self.__class__.__name__}("
            f"num_heads={self.num_heads}, "
            f"attn_dim={self.attn_dim}, "
            f"out_dim={self.out_dim}, "
            f"conv={self.conv})"
        )

    def reset_parameters(self) -> None:
        """Reset the parameters of the Attention layer"""
        self.gnn_q.reset_parameters()
        self.gnn_k.reset_parameters()
        self.gnn_v.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        flags: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Attention layer. Returns the value and attention matrix.

        Args:
            x (torch.Tensor): node features
            adj (torch.Tensor): adjacency matrix
            flags (torch.Tensor): node flags
            attention_mask (Optional[torch.Tensor], optional): attention mask for the attention matrix. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: value and attention matrix
        """

        if self.conv == "GCN":
            Q = self.gnn_q(x, adj)
            K = self.gnn_k(x, adj)
        else:
            Q = self.gnn_q(x)
            K = self.gnn_k(x)

        V = self.gnn_v(x, adj)
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

        # A: (B x num_heads) x N x N
        A = A.view(-1, *adj.shape)
        A = A.mean(dim=0)
        A = (A + A.transpose(-1, -2)) / 2  # symmetrize the attention matrix

        return V, A

    def get_gnn(
        self, in_dim: int, attn_dim: int, out_dim: int, conv: str = "GCN"
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
        """Initialize the three GNNs

        Args:
            in_dim (int): input dimension
            attn_dim (int): attention dimension
            out_dim (int): output dimension
            conv (str, optional): type of convolutional layer, choose from [GCN, MLP].
                Defaults to "GCN".

        Raises:
            NotImplementedError: raise an error if the convolutional layer is not implemented

        Returns:
            Tuple[Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]], Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]: three GNNs, one for the query, one for the key, and one for the value
        """

        if conv == "GCN":
            gnn_q = DenseGCNConv(in_dim, attn_dim)
            gnn_k = DenseGCNConv(in_dim, attn_dim)
            gnn_v = DenseGCNConv(in_dim, out_dim)

            return gnn_q, gnn_k, gnn_v

        elif conv == "MLP":
            num_layers = 2
            gnn_q = MLP(
                num_layers, in_dim, 2 * attn_dim, attn_dim, activate_func=torch.tanh
            )
            gnn_k = MLP(
                num_layers, in_dim, 2 * attn_dim, attn_dim, activate_func=torch.tanh
            )
            gnn_v = DenseGCNConv(in_dim, out_dim)

            return gnn_q, gnn_k, gnn_v

        else:
            raise NotImplementedError(f"Convolution layer {conv} not implemented.")


class AttentionLayer(torch.nn.Module):
    """AttentionLayer for ScoreNetworkA"""

    def __init__(
        self,
        num_linears: int,
        conv_input_dim: int,
        attn_dim: int,
        conv_output_dim: int,
        input_dim: int,
        output_dim: int,
        num_heads: int = 4,
        conv: str = "GCN",
        use_bn: bool = False,
    ) -> None:
        """Initialize the AttentionLayer

        Args:
            num_linears (int): number of linear layers in the MLP
            conv_input_dim (int): input dimension of the convolutional layer
            attn_dim (int): attention dimension
            conv_output_dim (int): output dimension of the convolutional layer
            input_dim (int): input dimension of the AttentionLayer (number of Attention)
            output_dim (int): output dimension of the MLP
            num_heads (int, optional): number of heads for the Attention. Defaults to 4.
            conv (str, optional): type of convolutional layer, choose from [GCN, MLP].
                Defaults to "GCN".
            use_bn (bool, optional): whether to use batch normalization in the MLP. Defaults to False.
        """

        super(AttentionLayer, self).__init__()

        self.attn = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.attn_dim = attn_dim
            self.attn.append(
                Attention(
                    conv_input_dim,
                    self.attn_dim,
                    conv_output_dim,
                    num_heads=num_heads,
                    conv=conv,
                )
            )

        self.hidden_dim = 2 * max(input_dim, output_dim)
        self.mlp = MLP(
            num_linears,
            2 * input_dim,
            self.hidden_dim,
            output_dim,
            use_bn=use_bn,
            activate_func=F.elu,
        )
        self.multi_channel = MLP(
            2,
            input_dim * conv_output_dim,
            self.hidden_dim,
            conv_output_dim,
            use_bn=use_bn,
            activate_func=F.elu,
        )

        # Reset the parameters
        self.reset_parameters()

    def __repr__(self) -> str:
        """Representation of the AttentionLayer

        Returns:
            str: representation of the AttentionLayer
        """

        return f"{self.__class__.__name__}({self.attn_dim}, {self.hidden_dim}, {self.attn_dim})"

    def reset_parameters(self) -> None:
        """Reset the parameters of the AttentionLayer"""
        # Reset the MLPs
        self.mlp.reset_parameters()
        self.multi_channel.reset_parameters()
        # Reset the attention layers
        for attn in self.attn:
            attn.reset_parameters()

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, flags: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the AttentionLayer. Returns a node feature matrix and an adjacency matrix.

        Args:
            x (torch.Tensor): node feature matrix (B x N x F_i)
                F_i is the input node feature dimension (=input_dim in GCNConv)
            adj (torch.Tensor): adjacency matrix (B x C_i x N x N)
            flags (Optional[torch.Tensor]): flags for the nodes

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: node feature matrix and adjacency matrix (B x N x F_o), (B x C_o x N x N)
                F_o is the output node feature dimension (=output_dim in GCNConv)
        """

        mask_list = []
        x_list = []
        for k in range(len(self.attn)):
            _x, mask = self.attn[k](x, adj[:, k, :, :], flags)
            mask_list.append(mask.unsqueeze(-1))
            x_list.append(_x)
        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)), flags)
        x_out = torch.tanh(x_out)

        mlp_in = torch.cat(
            [torch.cat(mask_list, dim=-1), adj.permute(0, 2, 3, 1)], dim=-1
        )
        shape = mlp_in.shape
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0, 3, 1, 2)
        _adj = _adj + _adj.transpose(-1, -2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out

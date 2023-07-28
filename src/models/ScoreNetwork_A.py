#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ScoreNetwork_A.py: ScoreNetworkA and BaselineNetwork classes.
These are ScoreNetwork models for the adjacency matrix A.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from src.models.layers import DenseGCNConv, MLP
from src.utils.graph_utils import (
    mask_adjs,
    pow_tensor,
    mask_x,
    node_feature_to_matrix,
)
from src.utils.cc_utils import default_mask
from src.models.attention import AttentionLayer


class BaselineNetworkLayer(torch.nn.Module):
    """BaselineNetworkLayer that operates on tensors derived from an adjacency matrix A.
    Used in the BaselineNetwork model.
    """

    def __init__(
        self,
        num_linears: int,
        conv_input_dim: int,
        conv_output_dim: int,
        input_dim: int,
        output_dim: int,
        use_bn: bool = False,
    ) -> None:
        """Initialize the BaselineNetworkLayer.

        Args:
            num_linears (int): number of linear layers in the MLP (except the first one)
            conv_input_dim (int): input dimension of the DenseGCNConv layers
            conv_output_dim (int): output dimension of the DenseGCNConv layers
            input_dim (int): number of DenseGCNConv layers (part of the input dimension of the final MLP)
            output_dim (int): output dimension of the final MLP
            use_bn (bool, optional): whether to use batch normalization in the MLP. Defaults to False.
        """
        super(BaselineNetworkLayer, self).__init__()

        # Initialize the parameters and the layers
        self.use_bn = use_bn
        self.convs = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.convs.append(DenseGCNConv(conv_input_dim, conv_output_dim))
        self.hidden_dim = max(input_dim, output_dim)
        self.mlp_in_dim = input_dim + 2 * conv_output_dim
        self.mlp = MLP(
            num_linears,
            self.mlp_in_dim,
            self.hidden_dim,
            output_dim,
            use_bn=self.use_bn,
            activate_func=F.elu,
        )
        self.multi_channel = MLP(
            2,
            input_dim * conv_output_dim,
            self.hidden_dim,
            conv_output_dim,
            use_bn=self.use_bn,
            activate_func=F.elu,
        )

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, flags: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the BaselineNetworkLayer.

        Args:
            x (torch.Tensor): node feature matrix
            adj (torch.Tensor): adjacency matrix
            flags (Optional[torch.Tensor]): optional flags for the node features

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: node feature matrix and adjacency matrix
        """

        # Apply all the DenseGCNConv layers
        x_list = []
        for k in range(len(self.convs)):
            _x = self.convs[k](x, adj[:, k, :, :])
            x_list.append(_x)
        # Concatenate the outputs of the DenseGCNConv layers, apply the multi-channel MLP, and mask the output
        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)), flags)
        x_out = torch.tanh(x_out)

        # Convert the node feature matrix to a node feature adjacency tensor
        x_matrix = node_feature_to_matrix(x_out)
        # Concatenate the node feature adjacency tensor with the original adjacency tensor
        mlp_in = torch.cat([x_matrix, adj.permute(0, 2, 3, 1)], dim=-1)
        shape = mlp_in.shape
        # Apply the final MLP on the concatenated adjacency tensor
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0, 3, 1, 2)
        _adj = _adj + _adj.transpose(-1, -2)
        # Mask the adjacency tensor
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out


class BaselineNetwork(torch.nn.Module):
    """BaselineNetwork to calculate the score with respect to the adjacency matrix A."""

    def __init__(
        self,
        max_feat_num: int,
        max_node_num: int,
        nhid: int,
        num_layers: int,
        num_linears: int,
        c_init: int,
        c_hid: int,
        c_final: int,
        adim: int,
        num_heads: int = 4,
        conv: str = "GCN",
        use_bn: bool = False,
        is_cc: bool = False,
    ) -> None:
        """Initialize the BaselineNetwork.

        Args:
            max_feat_num (int): maximum number of node features
            max_node_num (int): maximum number of nodes in the graphs
            nhid (int): number of hidden units in BaselineNetworkLayer layers
            num_layers (int): number of BaselineNetworkLayer layers
            num_linears (int): number of linear layers in the MLP of each BaselineNetworkLayer
            c_init (int): input dimension of the BaselineNetworkLayer (number of DenseGCNConv)
                Also the number of power iterations to "duplicate" the adjacency matrix
                as an input
            c_hid (int): number of hidden units in the MLP of each BaselineNetworkLayer
            c_final (int): output dimension of the MLP of the last BaselineNetworkLayer
            adim (int): UNUSED HERE. attention dimension (except for the first layer).
            num_heads (int, optional): UNUSED HERE. number of heads for the Attention. Defaults to 4.
            conv (str, optional): UNUSED HERE. type of convolutional layer, choose from [GCN, MLP]. Defaults to "GCN".
            use_bn (bool, optional): whether to use batch normalization in the MLP and the BaselineNetworkLayer(s). Defaults to False.
            is_cc (bool, optional): True if we generate combinatorial complexes. Defaults to False.
        """

        super(BaselineNetwork, self).__init__()

        # Initialize the parameters
        self.nfeat = max_feat_num
        self.max_node_num = max_node_num
        self.nhid = nhid
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final
        self.use_bn = use_bn
        self.is_cc = is_cc

        # Initialize the layers
        self.layers = torch.nn.ModuleList()
        for k in range(self.num_layers):
            if not (k):  # first layer
                self.layers.append(
                    BaselineNetworkLayer(
                        self.num_linears,
                        self.nfeat,
                        self.nhid,
                        self.c_init,
                        self.c_hid,
                        self.use_bn,
                    )
                )

            elif k == (self.num_layers - 1):  # last layer
                self.layers.append(
                    BaselineNetworkLayer(
                        self.num_linears,
                        self.nhid,
                        self.nhid,
                        self.c_hid,
                        self.c_final,
                        self.use_bn,
                    )
                )

            else:  # intermediate layers
                self.layers.append(
                    BaselineNetworkLayer(
                        self.num_linears,
                        self.nhid,
                        self.nhid,
                        self.c_hid,
                        self.c_hid,
                        self.use_bn,
                    )
                )

        # Initialize the final MLP
        self.fdim = self.c_hid * (self.num_layers - 1) + self.c_final + self.c_init
        self.final = MLP(
            num_layers=3,
            input_dim=self.fdim,
            hidden_dim=2 * self.fdim,
            output_dim=1,
            use_bn=self.use_bn,
            activate_func=F.elu,
        )
        # Initialize the mask
        self.mask = default_mask(self.max_node_num)
        self.mask.unsqueeze_(0)

        # Pick the right forward function
        if not (self.is_cc):
            self.forward = self.forward_graph
        else:
            self.forward = self.forward_cc

    def forward_graph(
        self, x: torch.Tensor, adj: torch.Tensor, flags: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the BaselineNetwork. Returns the score with respect to the adjacency matrix A.

        Args:
            x (torch.Tensor): node feature matrix
            adj (torch.Tensor): adjacency matrix
            flags (Optional[torch.Tensor], optional): optional flags for the score. Defaults to None.

        Returns:
            torch.Tensor: score with respect to the adjacency matrix A
        """

        # Duplicate the adjacency matrix as an input by creating power tensors
        adjc = pow_tensor(adj, self.c_init)

        # Apply all the BaselineNetworkLayer layers
        adj_list = [adjc]
        for k in range(self.num_layers):
            x, adjc = self.layers[k](x, adjc, flags)
            adj_list.append(adjc)

        # Concatenate the output of the BaselineNetworkLayer layers (B x N x N x (c_init + c_hid * (num_layers - 1) + c_final)
        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)
        out_shape = adjs.shape[:-1]  # B x N x N
        # Apply the final MLP on the concatenated adjacency tensor to compute the score
        score = self.final(adjs).view(*out_shape)

        # Mask the score
        self.mask = self.mask.to(score.device)
        score = score * self.mask

        # Mask the score with respect to the flags
        score = mask_adjs(score, flags)

        return score

    def forward_cc(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        rank2: torch.Tensor,
        flags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the BaselineNetwork. Returns the score with respect to the adjacency matrix A.

        Args:
            x (torch.Tensor): node feature matrix
            adj (torch.Tensor): adjacency matrix
            rank2 (torch.Tensor): rank2 incidence matrix
            flags (Optional[torch.Tensor], optional): optional flags for the score. Defaults to None.

        Returns:
            torch.Tensor: score with respect to the adjacency matrix A
        """
        return self.forward_graph(x, adj, flags)


class ScoreNetworkA(BaselineNetwork):
    """ScoreNetworkA to calculate the score with respect to the adjacency matrix A."""

    def __init__(
        self,
        max_feat_num: int,
        max_node_num: int,
        nhid: int,
        num_layers: int,
        num_linears: int,
        c_init: int,
        c_hid: int,
        c_final: int,
        adim: int,
        num_heads: int = 4,
        conv: str = "GCN",
        use_bn: bool = False,
        is_cc: bool = False,
    ) -> None:
        """Initialize the ScoreNetworkA model.

        Args:
            max_feat_num (int): maximum number of node features
            max_node_num (int): maximum number of nodes in the graphs
            nhid (int): number of hidden units in AttentionLayer layers
            num_layers (int): number of AttentionLayer layers
            num_linears (int): number of linear layers in the MLP of each AttentionLayer
            c_init (int): input dimension of the AttentionLayer (number of DenseGCNConv)
                Also the number of power iterations to "duplicate" the adjacency matrix
                as an input
            c_hid (int): number of hidden units in the MLP of each AttentionLayer
            c_final (int): output dimension of the MLP of the last AttentionLayer
            adim (int): attention dimension (except for the first layer).
            num_heads (int, optional): number of heads for the Attention. Defaults to 4.
            conv (str, optional): type of convolutional layer, choose from [GCN, MLP]. Defaults to "GCN".
            use_bn (bool, optional): whether to use batch normalization in the MLP and the AttentionLayer(s). Defaults to False.
            is_cc (bool, optional): True if we generate combinatorial complexes. Defaults to False.
        """

        super(ScoreNetworkA, self).__init__(
            max_feat_num,
            max_node_num,
            nhid,
            num_layers,
            num_linears,
            c_init,
            c_hid,
            c_final,
            adim,
            num_heads=4,
            conv="GCN",
            use_bn=use_bn,
            is_cc=is_cc,
        )

        # Initialize the parameters
        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv
        self.use_bn = use_bn
        self.is_cc = is_cc

        # Initialize the layers
        self.layers = torch.nn.ModuleList()
        for k in range(self.num_layers):
            if not (k):  # first layer
                self.layers.append(
                    AttentionLayer(
                        self.num_linears,
                        self.nfeat,
                        self.nhid,
                        self.nhid,
                        self.c_init,
                        self.c_hid,
                        self.num_heads,
                        self.conv,
                        self.use_bn,
                    )
                )
            elif k == (self.num_layers - 1):  # last layer
                self.layers.append(
                    AttentionLayer(
                        self.num_linears,
                        self.nhid,
                        self.adim,
                        self.nhid,
                        self.c_hid,
                        self.c_final,
                        self.num_heads,
                        self.conv,
                        self.use_bn,
                    )
                )
            else:  # intermediate layers
                self.layers.append(
                    AttentionLayer(
                        self.num_linears,
                        self.nhid,
                        self.adim,
                        self.nhid,
                        self.c_hid,
                        self.c_hid,
                        self.num_heads,
                        self.conv,
                        self.use_bn,
                    )
                )

        # Initialize the final MLP
        self.fdim = self.c_hid * (self.num_layers - 1) + self.c_final + self.c_init
        self.final = MLP(
            num_layers=3,
            input_dim=self.fdim,
            hidden_dim=2 * self.fdim,
            output_dim=1,
            use_bn=self.use_bn,
            activate_func=F.elu,
        )
        # Initialize the mask
        self.mask = default_mask(self.max_node_num)
        self.mask.unsqueeze_(0)

        # Pick the right forward function
        if not (self.is_cc):
            self.forward = self.forward_graph
        else:
            self.forward = self.forward_cc

    def forward_graph(
        self, x: torch.Tensor, adj: torch.Tensor, flags: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the ScoreNetworkA. Returns the score with respect to the adjacency matrix A.

        Args:
            x (torch.Tensor): node feature matrix
            adj (torch.Tensor): adjacency matrix
            flags (Optional[torch.Tensor], optional): optional flags for the score. Defaults to None.

        Returns:
            torch.Tensor: score with respect to the adjacency matrix A
        """

        # Duplicate the adjacency matrix as an input by creating power tensors
        adjc = pow_tensor(adj, self.c_init)

        # Apply all the AttentionLayer layers
        adj_list = [adjc]
        for k in range(self.num_layers):
            x, adjc = self.layers[k](x, adjc, flags)
            adj_list.append(adjc)

        # Concatenate the output of the AttentionLayer layers (B x N x N x (c_init + c_hid * (num_layers - 1) + c_final)
        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)
        out_shape = adjs.shape[:-1]  # B x N x N
        # Apply the final MLP on the concatenated adjacency tensor to compute the score
        score = self.final(adjs).view(*out_shape)

        # Mask the score
        self.mask = self.mask.to(score.device)
        score = score * self.mask

        # Mask the score with respect to the flags
        score = mask_adjs(score, flags)

        return score

    def forward_cc(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        rank2: torch.Tensor,
        flags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the ScoreNetworkA. Returns the score with respect to the adjacency matrix A.

        Args:
            x (torch.Tensor): node feature matrix
            adj (torch.Tensor): adjacency matrix
            rank2 (torch.Tensor): rank2 incidence matrix
            flags (Optional[torch.Tensor], optional): optional flags for the score. Defaults to None.

        Returns:
            torch.Tensor: score with respect to the adjacency matrix A
        """
        return self.forward_graph(x, adj, flags)

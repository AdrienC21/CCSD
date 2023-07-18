#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ScoreNetwork_X.py: ScoreNetworkX and ScoreNetworkX_GMH classes.
These are ScoreNetwork models for the node feature matrix X.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from src.models.layers import DenseGCNConv, MLP
from src.utils.graph_utils import mask_x, pow_tensor
from src.models.attention import AttentionLayer


# TODO: MODIFY THIS CLASS TO INCORPORATE RANK2 INCIDENCE MATRIX OPTIONALLY


class ScoreNetworkX(torch.nn.Module):
    """ScoreNetworkX network model.
    Returns the score with respect to the node feature matrix X."""

    def __init__(self, max_feat_num: int, depth: int, nhid: int) -> None:
        """Initialize ScoreNetworkX.

        Args:
            max_feat_num (int): maximum number of node features (input and output dimension of the network)
            depth (int): number of DenseGCNConv layers
            nhid (int): number of hidden units in DenseGCNConv layers
        """

        super(ScoreNetworkX, self).__init__()

        # Initialize parameters
        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid

        # Initialize DenseGCNConv layers
        self.layers = torch.nn.ModuleList()
        for k in range(self.depth):
            if not (k):  # first layer
                self.layers.append(DenseGCNConv(self.nfeat, self.nhid))
            else:  # other layers
                self.layers.append(DenseGCNConv(self.nhid, self.nhid))

        # Final layer is an MLP on the concatenation of all layers' outputs
        self.fdim = self.nfeat + self.depth * self.nhid
        self.final = MLP(
            num_layers=3,
            input_dim=self.fdim,
            hidden_dim=2 * self.fdim,
            output_dim=self.nfeat,
            use_bn=False,
            activate_func=F.elu,
        )

        # Initialize activation function
        self.activation = torch.tanh

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, flags: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass of the ScoreNetworkX model.

        Args:
            x (torch.Tensor): node feature matrix (B x N x F)
            adj (torch.Tensor): adjacency matrix (B x N x N)
            flags (Optional[torch.Tensor]): optional mask matrix (B x N x 1)

        Returns:
            torch.Tensor: score with respect to the node feature matrix (B x N x F)
        """

        # Apply all the DenseGCN layers
        x_list = [x]
        for k in range(self.depth):
            x = self.layers[k](x, adj)
            x = self.activation(x)
            x_list.append(x)

        # Concatenate all the layers' outputs (B x N x (F + num_layers x H))
        # B batch size, N max number of nodes, F number of features, H number of hidden units
        xs = torch.cat(x_list, dim=-1)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        # Apply the final MLP layer
        x = self.final(xs).view(*out_shape)

        # Mask the output
        x = mask_x(x, flags)

        return x


class ScoreNetworkX_GMH(torch.nn.Module):
    """ScoreNetworkX network model.
    Returns the score with respect to the node feature matrix X.
    """

    def __init__(
        self,
        max_feat_num: int,
        depth: int,
        nhid: int,
        num_linears: int,
        c_init: int,
        c_hid: int,
        c_final: int,
        adim: int,
        num_heads: int = 4,
        conv: str = "GCN",
    ) -> None:
        """Initialize ScoreNetworkX_GMH.

        Args:
            max_feat_num (int): maximum number of node features (input and output dimension of the network)
            depth (int): number of DenseGCNConv layers
            nhid (int): number of hidden units in DenseGCNConv layers
            num_linears (int): number of linear layers in AttentionLayer
            c_init (int): input dimension of the AttentionLayer (number of attention)
                Also the number of power iterations to "duplicate" the adjacency matrix
                as an input
            c_hid (int): output dimension of the MLP in the AttentionLayer
            c_final (int): output dimension of the MLP in the AttentionLayer for the last layer of this model
            adim (int): attention dimension (except for the first layer)
            num_heads (int, optional): number of heads for the Attention. Defaults to 4.
            conv (str, optional): type of convolutional layer, choose from [GCN, MLP].
                Defaults to "GCN".
        """
        super().__init__()

        # Initialize parameters
        self.depth = depth
        self.c_init = c_init

        # Initialize AttentionLayer layers
        self.layers = torch.nn.ModuleList()
        for k in range(self.depth):
            if not (k):  # first layer
                self.layers.append(
                    AttentionLayer(
                        num_linears,
                        max_feat_num,
                        nhid,
                        nhid,
                        c_init,
                        c_hid,
                        num_heads,
                        conv,
                    )
                )
            elif k == (self.depth - 1):  # last layer
                self.layers.append(
                    AttentionLayer(
                        num_linears, nhid, adim, nhid, c_hid, c_final, num_heads, conv
                    )
                )
            else:  # other layers
                self.layers.append(
                    AttentionLayer(
                        num_linears, nhid, adim, nhid, c_hid, c_hid, num_heads, conv
                    )
                )

        # Final layer is an MLP on the concatenation of all layers' outputs
        fdim = max_feat_num + depth * nhid
        self.final = MLP(
            num_layers=3,
            input_dim=fdim,
            hidden_dim=2 * fdim,
            output_dim=max_feat_num,
            use_bn=False,
            activate_func=F.elu,
        )

        # Initialize activation function
        self.activation = torch.tanh

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, flags: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass of the ScoreNetworkX_GMH model.

        Args:
            x (torch.Tensor): node feature matrix (B x N x F)
            adj (torch.Tensor): adjacency matrix (B x N x N)
            flags (Optional[torch.Tensor]): optional mask matrix (B x N x 1)

        Returns:
            torch.Tensor: score with respect to the node feature matrix (B x N x F)
        """

        # Duplicate the adjacency matrix as an input by creating power tensors
        adjc = pow_tensor(adj, self.c_init)

        # Apply all the AttentionLayer layers
        x_list = [x]
        for k in range(self.depth):
            x, adjc = self.layers[k](x, adjc, flags)
            x = self.activation(x)
            x_list.append(x)

        # Concatenate all the layers' outputs (B x N x (F + num_layers x H))
        # B batch size, N max number of nodes, F number of features, H number of hidden units
        xs = torch.cat(x_list, dim=-1)  # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)

        # Mask the output
        x = mask_x(x, flags)

        return x

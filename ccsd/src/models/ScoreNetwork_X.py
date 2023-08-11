#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ScoreNetwork_X.py: ScoreNetworkX and ScoreNetworkX_GMH classes.
These are ScoreNetwork models for the node feature matrix X.

Adapted from Jo, J. & al (2022)

Almost left untouched.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from ccsd.src.models.attention import AttentionLayer
from ccsd.src.models.layers import MLP, DenseGCNConv
from ccsd.src.utils.graph_utils import mask_x, pow_tensor


class ScoreNetworkX(torch.nn.Module):
    """ScoreNetworkX network model.
    Returns the score with respect to the node feature matrix X."""

    def __init__(
        self,
        max_feat_num: int,
        depth: int,
        nhid: int,
        use_bn: bool = False,
        is_cc: bool = False,
    ) -> None:
        """Initialize ScoreNetworkX.

        Args:
            max_feat_num (int): maximum number of node features (input and output dimension of the network)
            depth (int): number of DenseGCNConv layers
            nhid (int): number of hidden units in DenseGCNConv layers
            use_bn (bool, optional): True if we use batch normalization in the MLP. Defaults to False.
            is_cc (bool, optional): True if we generate combinatorial complexes. Defaults to False.
        """

        super(ScoreNetworkX, self).__init__()

        # Initialize parameters
        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid
        self.use_bn = use_bn
        self.is_cc = is_cc

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
            use_bn=self.use_bn,
            activate_func=F.elu,
        )

        # Initialize activation function
        self.activation = torch.tanh

        # Pick the right forward function
        if not (self.is_cc):
            self.forward = self.forward_graph
        else:
            self.forward = self.forward_cc

        # Reset the parameters
        self.reset_parameters()

    def __repr__(self) -> str:
        """String representation of the model.

        Returns:
            str: string representation of the model
        """

        return f"{self.__class__.__name__}(depth={self.depth}, nhid={self.nhid}, use_bn={self.use_bn}, is_cc={self.is_cc})"

    def reset_parameters(self) -> None:
        """Reset the parameters of the model."""

        # Reset the parameters of the DenseGCNConv layers
        for layer in self.layers:
            layer.reset_parameters()
        # Reset the parameters of the final MLP layer
        self.final.reset_parameters()

    def forward_graph(
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

    def forward_cc(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        rank2: torch.Tensor,
        flags: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass of the ScoreNetworkX model.

        Args:
            x (torch.Tensor): node feature matrix (B x N x F)
            adj (torch.Tensor): adjacency matrix (B x N x N)
            rank2 (torch.Tensor): rank2 incidence matrix (B x (NC2) x K)
            flags (Optional[torch.Tensor]): optional mask matrix (B x N x 1)

        Returns:
            torch.Tensor: score with respect to the node feature matrix (B x N x F)
        """
        return self.forward_graph(x, adj, flags)


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
        use_bn: bool = False,
        is_cc: bool = False,
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
            use_bn (bool, optional): True if we use batch normalization in the MLP and the AttentionLayer(s). Defaults to False.
            is_cc (bool, optional): True if we generate combinatorial complexes. Defaults to False.
        """
        super().__init__()

        # Initialize parameters
        self.depth = depth
        self.c_init = c_init
        self.use_bn = use_bn
        self.is_cc = is_cc

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
                        self.use_bn,
                    )
                )
            elif k == (self.depth - 1):  # last layer
                self.layers.append(
                    AttentionLayer(
                        num_linears,
                        nhid,
                        adim,
                        nhid,
                        c_hid,
                        c_final,
                        num_heads,
                        conv,
                        self.use_bn,
                    )
                )
            else:  # other layers
                self.layers.append(
                    AttentionLayer(
                        num_linears,
                        nhid,
                        adim,
                        nhid,
                        c_hid,
                        c_hid,
                        num_heads,
                        conv,
                        self.use_bn,
                    )
                )

        # Final layer is an MLP on the concatenation of all layers' outputs
        fdim = max_feat_num + depth * nhid
        self.final = MLP(
            num_layers=3,
            input_dim=fdim,
            hidden_dim=2 * fdim,
            output_dim=max_feat_num,
            use_bn=self.use_bn,
            activate_func=F.elu,
        )

        # Initialize activation function
        self.activation = torch.tanh

        # Pick the right forward function
        if not (self.is_cc):
            self.forward = self.forward_graph
        else:
            self.forward = self.forward_cc

        # Reset the parameters
        self.reset_parameters()

    def __repr__(self) -> str:
        """String representation of the ScoreNetworkX_GMH model.

        Returns:
            str: string representation
        """
        return f"{self.__class__.__name__}(depth={self.depth}, c_init={self.c_init}, use_bn={self.use_bn}, is_cc={self.is_cc})"

    def reset_parameters(self) -> None:
        """Reset the parameters of the ScoreNetworkX_GMH model."""
        # Reset the parameters of the AttentionLayer layers
        for attn in self.layers:
            attn.reset_parameters()
        # Reset the parameters of the final MLP
        self.final.reset_parameters()

    def forward_graph(
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

    def forward_cc(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        rank2: torch.Tensor,
        flags: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass of the ScoreNetworkX_GMH model.

        Args:
            x (torch.Tensor): node feature matrix (B x N x F)
            adj (torch.Tensor): adjacency matrix (B x N x N)
            rank2 (torch.Tensor): rank2 incidence matrix (B x (NC2) x K)
            flags (Optional[torch.Tensor]): optional mask matrix (B x N x 1)

        Returns:
            torch.Tensor: score with respect to the node feature matrix (B x N x F)
        """
        return self.forward_graph(x, adj, flags)

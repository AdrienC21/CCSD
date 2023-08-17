#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ScoreNetwork_A_CC.py: ScoreNetworkA_CC class.
This is a ScoreNetwork model for the adjacency matrix A in the higher-order domain.
"""

from typing import Optional

import torch
import torch.nn.functional as torch_func

from ccsd.src.models.attention import AttentionLayer
from ccsd.src.models.hodge_attention import HodgeAdjAttentionLayer
from ccsd.src.models.layers import MLP
from ccsd.src.utils.cc_utils import adj_to_hodgedual, default_mask, hodgedual_to_adj
from ccsd.src.utils.graph_utils import mask_adjs, pow_tensor


class ScoreNetworkA_CC(torch.nn.Module):
    """ScoreNetworkA_CC to calculate the score with respect to the adjacency matrix A in the higher-order domain."""

    def __init__(
        self,
        max_feat_num: int,
        max_node_num: int,
        d_min: int,
        d_max: int,
        nhid: int,
        nhid_h: int,
        num_layers: int,
        num_layers_h: int,
        num_linears: int,
        num_linears_h: int,
        c_init: int,
        c_hid: int,
        c_hid_h: int,
        c_final: int,
        c_final_h: int,
        adim: int,
        adim_h: int,
        num_heads: int = 4,
        num_heads_h: int = 4,
        conv: str = "GCN",
        conv_hodge: str = "HCN",
        use_bn: bool = False,
        is_cc: bool = True,
    ) -> None:
        """Initialize the ScoreNetworkA_CC model.

        Args:
            max_feat_num (int): maximum number of node features
            max_node_num (int): maximum number of nodes in the graphs
            d_min (int): minimum dimension of the rank-2 cells
            d_max (int): maximum dimension of the rank-2 cells
            nhid (int): number of hidden units in AttentionLayer layers
            nhid_h (int): number of hidden units in HodgeAdjAttentionLayer layers
            num_layers (int): number of AttentionLayer layers
            num_layers_h (int): number of HodgeAdjAttentionLayer layers
            num_linears (int): number of linear layers in the MLP of each AttentionLayer
            num_linears_h (int): number of linear layers in the MLP of each HodgeAdjAttentionLayer
            c_init (int): input dimension of the AttentionLayer and the HodgeAdjAttentionLayer
                (number of DenseGCNConv and DenseHCNConv)
                Also the number of power iterations to "duplicate" the adjacency matrix
                as an input
            c_hid (int): number of hidden units in the MLP of each AttentionLayer
            c_hid_h (int): number of hidden units in the MLP of each HodgeAdjAttentionLayer
            c_final (int): output dimension of the MLP of the last AttentionLayer
            c_final_h (int): output dimension of the MLP of the last HodgeAdjAttentionLayer
            adim (int): attention dimension for the AttentionLayer (except for the first layer).
            adim_h (int): attention dimension for the HodgeAdjAttentionLayer (except for the first layer).
            num_heads (int, optional): number of heads for the Attention. Defaults to 4.
            num_heads_h (int, optional): number of heads for the HodgeAdjAttention. Defaults to 4.
            conv (str, optional): type of convolutional layer, choose from [HCN, MLP]. Defaults to "GCN".
            conv_hodge (str, optional): type of convolutional layer for the hodge layers, choose from [HCN, MLP]. Defaults to "HCN".
            use_bn (bool, optional): whether to use batch normalization in the MLP and the AttentionLayer(s). Defaults to False.
            is_cc (bool, optional): True if we generate combinatorial complexes. Defaults to True.
        """

        super(ScoreNetworkA_CC, self).__init__()

        # Initialize the parameters
        self.max_feat_num = max_feat_num
        self.max_node_num = max_node_num
        self.N = max_node_num
        self.d_min = d_min
        self.d_max = d_max
        self.nhid = nhid
        self.nhid_h = nhid_h
        self.num_layers = num_layers
        self.num_layers_h = num_layers_h
        self.num_linears = num_linears
        self.num_linears_h = num_linears_h
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_hid_h = c_hid_h
        self.c_final = c_final
        self.c_final_h = c_final_h
        self.adim = adim
        self.adim_h = adim_h
        self.num_heads = num_heads
        self.num_heads_h = num_heads_h
        self.conv = conv
        self.conv_hodge = conv_hodge
        self.use_bn = use_bn
        self.is_cc = is_cc

        # Initialize the layers
        # AttentionLayer layers
        self.layers = torch.nn.ModuleList()
        for k in range(self.num_layers):
            if not (k):  # first layer
                self.layers.append(
                    AttentionLayer(
                        self.num_linears,
                        self.max_feat_num,
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
        # HodgeAdjAttentionLayer layers
        self.layers_hodge = torch.nn.ModuleList()
        for k in range(self.num_layers_h):
            if not (k):  # first layer
                self.layers_hodge.append(
                    HodgeAdjAttentionLayer(
                        self.num_linears_h,
                        self.c_init,
                        self.nhid_h,
                        self.c_hid_h,
                        self.N,
                        self.d_min,
                        self.d_max,
                        self.num_heads_h,
                        self.conv_hodge,
                        self.use_bn,
                    )
                )
            elif k == (self.num_layers_h - 1):  # last layer
                self.layers_hodge.append(
                    HodgeAdjAttentionLayer(
                        self.num_linears_h,
                        self.c_hid_h,
                        self.adim_h,
                        self.c_final_h,
                        self.N,
                        self.d_min,
                        self.d_max,
                        self.num_heads_h,
                        self.conv_hodge,
                        self.use_bn,
                    )
                )
            else:  # intermediate layers
                self.layers_hodge.append(
                    HodgeAdjAttentionLayer(
                        self.num_linears_h,
                        self.c_hid_h,
                        self.adim_h,
                        self.c_hid_h,
                        self.N,
                        self.d_min,
                        self.d_max,
                        self.num_heads_h,
                        self.conv_hodge,
                        self.use_bn,
                    )
                )

        # Initialize the final MLP
        self.fdim = (
            self.c_hid * (self.num_layers - 1)
            + self.c_final
            + self.c_init
            + self.c_hid_h * (self.num_layers_h - 1)
            + self.c_final_h
            + self.c_init
        )
        self.final = MLP(
            num_layers=3,
            input_dim=self.fdim,
            hidden_dim=2 * self.fdim,
            output_dim=1,
            use_bn=self.use_bn,
            activate_func=torch_func.elu,
        )
        # Initialize the mask
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.mask = default_mask(self.max_node_num, device)
        self.mask.unsqueeze_(0)

        # Pick the right forward function
        if not (self.is_cc):
            raise ValueError("ScoreNetworkA_CC is only for combinatorial complexes")

        # Reset the parameters
        self.reset_parameters()

    def __repr__(self) -> str:
        """Representation of the ScoreNetworkA model.

        Returns:
            str: representation of the ScoreNetworkA model
        """
        return (
            f"{self.__class__.__name__}("
            f"max_feat_num={self.max_feat_num}, "
            f"max_node_num (N)={self.max_node_num}, "
            f"nhid={self.nhid}, "
            f"num_layers={self.num_layers}, "
            f"num_linears={self.num_linears}, "
            f"c_init={self.c_init}, "
            f"c_hid={self.c_hid}, "
            f"c_final={self.c_final}, "
            f"adim={self.adim}, "
            f"num_heads={self.num_heads}, "
            f"conv={self.conv}, "
            f"conv_hodge={self.conv_hodge}, "
            f"use_bn={self.use_bn}, "
            f"is_cc={self.is_cc})"
        )

    def reset_parameters(self) -> None:
        """Reset the parameters of the model."""
        # Reset the parameters of the AttentionLayer layers
        for attn in self.layers:
            attn.reset_parameters()
        # Reset the parameters of the final MLP
        self.final.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        rank2: torch.Tensor,
        flags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the ScoreNetworkA_CC. Returns the score with respect to the adjacency matrix A.

        Args:
            x (torch.Tensor): node feature matrix
            adj (torch.Tensor): adjacency matrix
            rank2 (torch.Tensor): rank-2 incidence matrix
            flags (Optional[torch.Tensor], optional): optional flags for the score. Defaults to None.

        Returns:
            torch.Tensor: score with respect to the adjacency matrix A
        """

        # Duplicate the adjacency matrix as an input by creating power tensors
        adjc = pow_tensor(adj, self.c_init)
        hodge_adjc = adj_to_hodgedual(adjc)  # get hodge dual of adjc
        # Apply all the AttentionLayer layers
        adj_list = [adjc]
        _x = x.clone()
        for k in range(self.num_layers):
            _x, adjc = self.layers[k](_x, adjc, flags)
            adj_list.append(adjc)
        # Apply all the HodgeAdjAttentionLayer layers
        hodge_adj_list = [hodge_adjc]
        _rank2 = rank2.clone()
        for k in range(self.num_layers_h):
            hodge_adjc, _rank2 = self.layers_hodge[k](hodge_adjc, _rank2, flags)
            hodge_adj_list.append(hodge_adjc)

        # Concatenate the output of the AttentionLayer layers (B x N x N x (c_init + c_hid * (num_layers - 1) + c_final)
        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)
        out_shape = adjs.shape[:-1]  # B x N x N
        # Concatenate the output of the HodgeAdjAttentionLayer layers (B x (NC2) x (NC2) x (c_init + c_hid * (num_layers_h - 1) + c_final_h)
        hodge_adjs = torch.cat(hodge_adj_list, dim=1)
        adj_hodge_adjs = hodgedual_to_adj(hodge_adjs).permute(
            0, 2, 3, 1
        )  # (B x N x N x (c_init + c_hid_h * (num_layers_h - 1) + c_final_h)
        # Concatenate the two outputs
        out = torch.cat(
            [adjs, adj_hodge_adjs], dim=-1
        )  # B x N x N x (c_init + c_hid * (num_layers - 1) + c_final + c_init + c_hid_h * (num_layers_h - 1) + c_final_h)
        # Apply the final MLP on the concatenated adjacency tensor to compute the score
        score = self.final(out).view(*out_shape)

        # Mask the score
        self.mask = self.mask.to(score.device)
        score = score * self.mask

        # Mask the score with respect to the flags
        score = mask_adjs(score, flags)

        return score

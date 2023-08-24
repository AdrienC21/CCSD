#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ScoreNetwork_F.py: ScoreNetworkF class.
This is a ScoreNetwork model that operates on the rank2 incidence matrix of the combinatorial complex.
"""

from typing import Optional

import torch
import torch.nn.functional as torch_func

from ccsd.src.models.hodge_layers import HodgeNetworkLayer
from ccsd.src.models.layers import MLP
from ccsd.src.utils.cc_utils import (
    default_mask,
    get_rank2_dim,
    mask_rank2,
    pow_tensor_cc,
)
from ccsd.src.utils.models_utils import get_ones


class ScoreNetworkF(torch.nn.Module):
    """ScoreNetworkF to calculate the score with respect to the rank2 incidence matrix."""

    def __init__(
        self,
        num_layers_mlp: int,
        num_layers: int,
        num_linears: int,
        nhid: int,
        c_hid: int,
        c_final: int,
        cnum: int,
        max_node_num: int,
        d_min: int,
        d_max: int,
        use_hodge_mask: bool = True,
        use_bn: bool = False,
        is_cc: bool = True,
    ) -> None:
        """Initialize the ScoreNetworkF model.

        Args:
            num_layers_mlp (int): number of layers in the final MLP
            num_layers (int): number of HodgeNetworkLayer layers
            num_linears (int): number of additional layers in the MLP of the HodgeNetworkLayer
            nhid (int): number of hidden units in the MLP of the HodgeNetworkLayer
            c_hid (int): number of output channels in the intermediate HodgeNetworkLayer(s)
            c_final (int): number of output channels in the last HodgeNetworkLayer
            cnum (int): number of power iterations of the rank2 incidence matrix
                Also number of input channels in the first HodgeNetworkLayer.
            max_node_num (int): maximum number of nodes in the combinatorial complex
            d_min (int): minimum size of the rank2 cells
            d_max (int): maximum size of the rank2 cells
            use_hodge_mask (bool, optional): whether to use the hodge mask. Defaults to False.
            use_bn (bool, optional): whether to use batch normalization in the MLP. Defaults to False.
            is_cc (bool, optional): whether the input is a combinatorial complex (ALWAYS THE CASE). Defaults to True.
        """
        super(ScoreNetworkF, self).__init__()

        # Initialize the parameters
        self.num_layers_mlp = num_layers_mlp
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.nhid = nhid
        self.c_hid = c_hid
        self.c_final = c_final
        self.cnum = cnum
        self.max_node_num = max_node_num
        self.d_min = d_min
        self.d_max = d_max
        self.use_hodge_mask = use_hodge_mask
        self.use_bn = use_bn
        self.is_cc = is_cc  # always the case

        # Get the rank2 dimension
        self.rows, self.cols = get_rank2_dim(self.max_node_num, self.d_min, self.d_max)

        # Initialize the layers
        self.layers = torch.nn.ModuleList()
        for k in range(self.num_layers):
            if not (k):  # first layer
                self.layers.append(
                    HodgeNetworkLayer(
                        self.num_linears,
                        self.cnum,
                        self.nhid,
                        self.c_hid,
                        self.d_min,
                        self.d_max,
                        self.use_bn,
                    )
                )

            elif k == (self.num_layers - 1):  # last layer
                self.layers.append(
                    HodgeNetworkLayer(
                        self.num_linears,
                        self.c_hid,
                        self.nhid,
                        self.c_final,
                        self.d_min,
                        self.d_max,
                        self.use_bn,
                    )
                )

            else:  # intermediate layers
                self.layers.append(
                    HodgeNetworkLayer(
                        self.num_linears,
                        self.c_hid,
                        self.nhid,
                        self.c_hid,
                        self.d_min,
                        self.d_max,
                        self.use_bn,
                    )
                )

        # Initialize the final MLP
        self.fdim = self.c_hid * (self.num_layers - 1) + self.c_final + self.cnum
        self.final = MLP(
            num_layers=self.num_layers_mlp,
            input_dim=self.fdim,
            hidden_dim=2 * self.fdim,
            output_dim=1,
            use_bn=self.use_bn,
            activate_func=torch_func.elu,
        )

        # Initialize the masks (hodge mask and score mask)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.hodge_mask = (
            default_mask(self.rows)
            if self.use_hodge_mask
            else get_ones((self.rows, self.rows), device)
        )
        self.hodge_mask.unsqueeze_(0)
        self.mask = get_ones((self.rows, self.cols), device)
        self.mask = self.mask.unsqueeze(0)

        # Initialize the parameters (glorot weights, zeros bias), default reset for batchnorm (if any)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the model."""
        # Reset the HodgeNetworkLayer layers
        for layer in self.layers:
            layer.reset_parameters()
        # Reset the final MLP
        self.final.reset_parameters()

    def __repr__(self) -> str:
        """Representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"num_layers_mlp={self.num_layers_mlp}, "
            f"num_layers={self.num_layers}, "
            f"num_linears={self.num_linears}, "
            f"nhid={self.nhid}, "
            f"c_hid={self.c_hid}, "
            f"c_final={self.c_final}, "
            f"cnum={self.cnum}, "
            f"max_node_num={self.max_node_num}, "
            f"d_min={self.d_min}, "
            f"d_max={self.d_max}, "
            f"use_hodge_mask={self.use_hodge_mask}, "
            f"use_bn={self.use_bn}, "
            f"is_cc={self.is_cc})"
        )

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
        # Calculate the power iteration of the rank2 incidence matrix using the Hodge Laplacian
        rank2c = pow_tensor_cc(rank2, self.cnum, self.hodge_mask)

        # Apply all the HodgeNetworkLayer layers
        rank2_list = [rank2c]
        _rank2c = rank2c.clone()
        for k in range(self.num_layers):
            _rank2c = self.layers[k](_rank2c, self.max_node_num, flags)
            rank2_list.append(_rank2c)

        # Concatenate the output of the HodgeNetworkLayer layers (B x (NC2) x K x (cnum + c_hid * (num_layers - 1) + c_final)
        rank2s = torch.cat(rank2_list, dim=1).permute(0, 2, 3, 1)
        out_shape = rank2s.shape[:-1]  # B x (NC2) x K

        # Apply the final MLP on the concatenated rank2 incidence tensors to compute the score
        score = self.final(rank2s).view(*out_shape)

        # Mask the score
        self.mask = self.mask.to(score.device)
        score = score * self.mask

        # Mask the score with respect to the flags
        score = mask_rank2(score, self.max_node_num, self.d_min, self.d_max, flags)

        return score

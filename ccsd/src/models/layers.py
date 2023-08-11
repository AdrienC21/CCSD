#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""layers.py: DenseGCNConv and MLP class for the Attention layers/the ScoreNetwork models.
It also contains some parameters initialization functions.

Adapted from Jo, J. & al (2022)

Almost left untouched.
"""

import math
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch.nn import Parameter


def glorot(tensor: Optional[torch.Tensor]) -> None:
    """Initialize the tensor with Glorot uniform initialization.
    (Glorot uniform initialization is also called Xavier uniform initialization)

    Args:
        tensor (Optional[torch.Tensor]): tensor to be initialized
    """
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor: Optional[torch.Tensor]) -> None:
    """Initialize the tensor with zeros.

    Args:
        tensor (Optional[torch.Tensor]): tensor to be initialized
    """
    if tensor is not None:
        tensor.data.fill_(0)


def reset(value: Any) -> None:
    """Reset the parameters of a value object and all its children.
    The value object must have a `reset_parameters` method to reset its parameters
    and a `children` method that returns its children to also reset its children if any.

    Args:
        value (Any): value object with parameters to be reset
    """
    if hasattr(value, "reset_parameters"):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, "children") else []:
            reset(child)


class DenseGCNConv(torch.nn.Module):
    """Dense GCN layer (Graph Convolutional Network layer) with adjacency matrix.

    It is similar to the operator described in
    Kipf, T. N., & Welling, M. (2016), Semi-Supervised Classification with Graph Convolutional Networks

    See also torch geometric (torch_geometric.nn.conv.GCNConv)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        bias: bool = True,
    ) -> None:
        """Initialize the DenseGCNConv layer.

        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            improved (bool, optional): if True, put 2 in the diagonal coefficients of the adjacency matrix, else 1. Defaults to False.
            bias (bool, optional): if True, add bias parameters. Defaults to True.
        """
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        # Initialize the bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize the parameters (glorot for the weight and zeros for the bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the DenseGCNConv layer.
        Initialize them with Glorot uniform initialization for the weight and zeros for the bias.
        """
        glorot(self.weight)
        zeros(self.bias)

    def __repr__(self) -> str:
        """Return a string representation of the DenseGCNConv layer.

        Returns:
            str: string representation of the DenseGCNConv layer
        """
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        add_loop: bool = True,
    ) -> torch.Tensor:
        """Forward pass of the DenseGCNConv layer.

        Args:
            x (torch.Tensor): node feature matrix (B * N * F_i)
            adj (torch.Tensor): adjacency matrix (B * N * N)
            mask (Optional[torch.Tensor], optional): Optional mask for the output. Defaults to None.
            add_loop (bool, optional): if False, the layer will
                not automatically add self-loops to the adjacency matrices. Defaults to True.

        Returns:
            torch.Tensor: output of the DenseGCNConv layer (B * N * F_o)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        # Add self-loops to the adjacency matrix
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        # Add the bias
        if self.bias is not None:
            out = out + self.bias

        # Apply the mask
        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


class MLP(torch.nn.Module):
    """Multi-Layer Perceptron (MLP) layer."""

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_bn: bool = False,
        activate_func: Callable[[torch.Tensor], torch.Tensor] = F.relu,
    ) -> None:
        """Initialize the MLP layer.

        Args:
            num_layers (int): number of additional layers in the neural networks (so except the input layer).
                If num_layers=1, this reduces to a linear model.
            input_dim (int): input dimension
            hidden_dim (int): hidden dimension of the intermediate layers
            output_dim (int): output dimension
            use_bn (bool, optional): if True, add Batch Normalization after each hidden layer.
                Defaults to False.
            activate_func (Callable[[torch.Tensor], torch.Tensor], optional): activation layer
                (must be non-linear) to be applied at the end of each layer. Defaults to F.relu.

        Raises:
            ValueError: raise an error if the number of layers is not greater or equal to 1.
        """

        super(MLP, self).__init__()

        # Initialize the parameters
        self.linear_or_not = (
            True  # default is linear model, will be change if more than 1 layer
        )
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_bn = use_bn
        self.activate_func = activate_func

        if self.num_layers < 1:
            raise ValueError("Number of layers should be greater of equal to 1.")
        elif self.num_layers == 1:
            # Linear model
            self.linear = torch.nn.Linear(self.input_dim, self.output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            # Add initial layer
            self.linears.append(torch.nn.Linear(self.input_dim, self.hidden_dim))
            # Add hidden layers
            for _ in range(self.num_layers - 2):
                self.linears.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
            # Add final layer
            self.linears.append(torch.nn.Linear(self.hidden_dim, self.output_dim))

            # Add batch normalization layers
            if self.use_bn:
                self.batch_norms = torch.nn.ModuleList()
                for _ in range(self.num_layers - 1):
                    self.batch_norms.append(torch.nn.BatchNorm1d(self.hidden_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the MLP layer.
        Initialize them with Glorot uniform initialization for the weight and zeros for the bias.
        """
        if self.linear_or_not:
            # Linear model
            glorot(self.linear.weight)
            zeros(self.linear.bias)
        else:
            # MLP model
            for layer in range(self.num_layers - 1):
                # Reset the linear layer
                glorot(self.linears[layer].weight)
                zeros(self.linears[layer].bias)
                # Reset batch normalization layer
                if self.use_bn:
                    self.batch_norms[layer].reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP layer.

        Args:
            x (torch.Tensor): input tensor (num_classes * B * N * F_i)
                num_classes is the number of classes of input, to be treated with different weights and biases
                B is the batch size
                N is the maximum number of nodes across the batch
                F_i is the input node feature dimension (=input_dim)

        Returns:
            torch.Tensor: output tensor (num_classes * B * N * F_o)
                F_o is the output node feature dimension (=output_dim)
        """
        if self.linear_or_not:
            # Linear model
            return self.linear(x)
        else:
            # MLP model
            h = x
            for layer in range(self.num_layers - 1):
                # Apply the linear layer
                h = self.linears[layer](h)
                # Apply batch normalization
                if self.use_bn:
                    h = self.batch_norms[layer](h)
                # Apply the activation function
                h = self.activate_func(h)
            # Apply the final layer
            return self.linears[self.num_layers - 1](h)

    def __repr__(self) -> str:
        """Return a string representation of the MLP layer.

        Returns:
            str: string representation of the MLP layer
        """
        return "{}(layers={}, dim=({}, {}, {}), batch_norm={})".format(
            self.__class__.__name__,
            self.num_layers,
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
            self.use_bn,
        )

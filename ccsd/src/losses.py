#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""losses.py: Loss functions for training the SDEs.

Adapted from Jo, J. & al (2022), except for get_sde_loss_fn_cc, almost left untouched.
"""

from typing import Callable, Optional, Tuple

import torch

from ccsd.src.sde import SDE, VESDE, VPSDE, subVPSDE
from ccsd.src.utils.cc_utils import gen_noise_rank2, mask_rank2
from ccsd.src.utils.graph_utils import gen_noise, mask_adjs, mask_x, node_flags


def get_score_fn(
    sde: SDE, model: torch.nn.Module, train: bool = True, continuous: bool = True
) -> Callable[
    [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor
]:
    """Return the score function for the SDE and the model.

    Args:
        sde (SDE): Stochastic Differential Equation (SDE)
        model (torch.nn.Module): neural network model that predicts the score
        train (bool, optional): whether or not we train the model. Defaults to True.
        continuous (bool, optional): if the SDE is continuous (discrete NOT IMPLEMENTED HERE). Defaults to True.

    Raises:
        NotImplementedError: raise an error if the SDE is not implemented

    Returns:
        Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], float], torch.Tensor]: score function
    """

    if not (train):  # if not training, set model to eval mode
        model.eval()
    model_fn = model  # alias for model function

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
        # Scale neural network output by standard deviation and flip sign
        def score_fn(
            x: torch.Tensor,
            adj: torch.Tensor,
            flags: Optional[torch.Tensor],
            t: torch.Tensor,
        ) -> torch.Tensor:
            """Return the predicted score function by the model for the SDE and scale by
            the standard deviation computed via the marginal probability.

            Args:
                x (torch.Tensor): node features
                adj (torch.Tensor): adjacency matrix
                flags (Optional[torch.Tensor]): optional node flags
                t (torch.Tensor): tensor of random timesteps

            Raises:
                NotImplementedError: if discrete SDE, not implemented

            Returns:
                torch.Tensor: predicted scaled score function
            """
            if continuous:
                score = model_fn(x, adj, flags)
                std = sde.marginal_prob(torch.zeros_like(adj), t)[1]
            else:
                raise NotImplementedError(f"Discrete not supported")
            score = -score / std[:, None, None]
            return score

    elif isinstance(sde, VESDE):
        # Just return the neural network output
        def score_fn(
            x: torch.Tensor,
            adj: torch.Tensor,
            flags: Optional[torch.Tensor],
            t: torch.Tensor,
        ) -> torch.Tensor:
            """Return the predicted score function by the model for the SDE.

            Args:
                x (torch.Tensor): node features
                adj (torch.Tensor): adjacency matrix
                flags (Optional[torch.Tensor]): optional node flags
                t (torch.Tensor): tensor of random timesteps (UNUSED HERE)

            Raises:
                NotImplementedError: if discrete SDE, not implemented

            Returns:
                torch.Tensor: predicted score function
            """
            if continuous:
                score = model_fn(x, adj, flags)
            else:
                raise NotImplementedError(f"Discrete not supported")
            return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not supported.")

    return score_fn


def get_score_fn_cc(
    sde: SDE, model: torch.nn.Module, train: bool = True, continuous: bool = True
) -> Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
    torch.Tensor,
]:
    """Return the score function for the SDE and the model.

    Args:
        sde (SDE): Stochastic Differential Equation (SDE)
        model (torch.nn.Module): neural network model that predicts the score
        train (bool, optional): whether or not we train the model. Defaults to True.
        continuous (bool, optional): if the SDE is continuous (discrete NOT IMPLEMENTED HERE). Defaults to True.

    Raises:
        NotImplementedError: raise an error if the SDE is not implemented

    Returns:
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], float], torch.Tensor]: score function
    """

    if not (train):  # if not training, set model to eval mode
        model.eval()
    model_fn = model  # alias for model function

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
        # Scale neural network output by standard deviation and flip sign
        def score_fn(
            x: torch.Tensor,
            adj: torch.Tensor,
            rank2: torch.Tensor,
            flags: Optional[torch.Tensor],
            t: torch.Tensor,
        ) -> torch.Tensor:
            """Return the predicted score function by the model for the SDE and scale by
            the standard deviation computed via the marginal probability.

            Args:
                x (torch.Tensor): node features
                adj (torch.Tensor): adjacency matrix
                rank2 (torch.Tensor): rank2 incidence tensor
                flags (Optional[torch.Tensor]): optional node flags
                t (torch.Tensor): tensor of random timesteps

            Raises:
                NotImplementedError: if discrete SDE, not implemented

            Returns:
                torch.Tensor: predicted scaled score function
            """
            if continuous:
                score = model_fn(x, adj, rank2, flags)
                std = sde.marginal_prob(torch.zeros_like(adj), t)[1]
            else:
                raise NotImplementedError(f"Discrete not supported")
            score = -score / std[:, None, None]
            return score

    elif isinstance(sde, VESDE):
        # Just return the neural network output
        def score_fn(
            x: torch.Tensor,
            adj: torch.Tensor,
            rank2: torch.Tensor,
            flags: Optional[torch.Tensor],
            t: torch.Tensor,
        ) -> torch.Tensor:
            """Return the predicted score function by the model for the SDE.

            Args:
                x (torch.Tensor): node features
                adj (torch.Tensor): adjacency matrix
                rank2 (torch.Tensor): rank2 incidence tensor
                flags (Optional[torch.Tensor]): optional node flags
                t (torch.Tensor): tensor of random timesteps (UNUSED HERE)

            Raises:
                NotImplementedError: if discrete SDE, not implemented

            Returns:
                torch.Tensor: predicted score function
            """
            if continuous:
                score = model_fn(x, adj, rank2, flags)
            else:
                raise NotImplementedError(f"Discrete not supported")
            return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not supported.")

    return score_fn


def get_sde_loss_fn(
    sde_x: SDE,
    sde_adj: SDE,
    train: bool = True,
    reduce_mean: bool = False,
    continuous: bool = True,
    likelihood_weighting: bool = False,
    eps: float = 1e-5,
) -> Callable[
    [torch.nn.Module, torch.nn.Module, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]:
    """Return the loss function for the SDEs with specific parameters.

    Args:
        sde_x (SDE): SDE for node features
        sde_adj (SDE): SDE for adjacency matrix
        train (bool, optional): whether or not we are training the model. Defaults to True.
        reduce_mean (bool, optional): if True, we reduce the loss by first taking the mean along the last axis. Defaults to False.
        continuous (bool, optional): if the SDE is continuous. Defaults to True.
        likelihood_weighting (bool, optional): if True, weight the loss with standard deviations. Defaults to False.
        eps (float, optional): parameter for sampling time. Defaults to 1e-5.

    Returns:
        Callable[[torch.nn.Module, torch.nn.Module, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]: loss function
    """

    # Reduce operator for loss
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    # Loss function for SDEs
    def loss_fn(
        model_x: torch.nn.Module,
        model_adj: torch.nn.Module,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get score functions
        score_fn_x = get_score_fn(sde_x, model_x, train=train, continuous=continuous)
        score_fn_adj = get_score_fn(
            sde_adj, model_adj, train=train, continuous=continuous
        )

        # Sample time
        t = torch.rand(adj.shape[0], device=adj.device) * (sde_adj.T - eps) + eps
        # Get node flags from adjacency matrix
        flags = node_flags(adj)

        # Sample noise for the node features
        z_x = gen_noise(x, flags, sym=False)
        mean_x, std_x = sde_x.marginal_prob(x, t)
        # Perturb node features
        perturbed_x = mean_x + std_x[:, None, None] * z_x
        perturbed_x = mask_x(perturbed_x, flags)

        # Sample noise for the adjacency matrix
        z_adj = gen_noise(adj, flags, sym=True)
        mean_adj, std_adj = sde_adj.marginal_prob(adj, t)
        # Perturb adjacency matrix
        perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
        perturbed_adj = mask_adjs(perturbed_adj, flags)

        # Compute score functions
        score_x = score_fn_x(perturbed_x, perturbed_adj, flags, t)
        score_adj = score_fn_adj(perturbed_x, perturbed_adj, flags, t)

        # Compute losses
        if not (likelihood_weighting):
            losses_x = torch.square(score_x * std_x[:, None, None] + z_x)
            losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1)

            losses_adj = torch.square(score_adj * std_adj[:, None, None] + z_adj)
            losses_adj = reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1)

        else:
            g2_x = sde_x.sde(torch.zeros_like(x), t)[1] ** 2
            losses_x = torch.square(score_x + z_x / std_x[:, None, None])
            losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1) * g2_x

            g2_adj = sde_adj.sde(torch.zeros_like(adj), t)[1] ** 2
            losses_adj = torch.square(score_adj + z_adj / std_adj[:, None, None])
            losses_adj = (
                reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1) * g2_adj
            )

        return torch.mean(losses_x), torch.mean(losses_adj)

    return loss_fn


def get_sde_loss_fn_cc(
    sde_x: SDE,
    sde_adj: SDE,
    sde_rank2: SDE,
    d_min: int,
    d_max: int,
    train: bool = True,
    reduce_mean: bool = False,
    continuous: bool = True,
    likelihood_weighting: bool = False,
    eps: float = 1e-5,
) -> Callable[
    [
        torch.nn.Module,
        torch.nn.Module,
        torch.nn.Module,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Return the loss function for the SDEs with specific parameters.

    Args:
        sde_x (SDE): SDE for node features
        sde_adj (SDE): SDE for adjacency matrix
        sde_rank2 (SDE): SDE for rank-2 incidence tensor
        d_min (int): minimum size of the rank-2 cells
        d_max (int): maximum size of the rank-2 cells
        train (bool, optional): whether or not we are training the model. Defaults to True.
        reduce_mean (bool, optional): if True, we reduce the loss by first taking the mean along the last axis. Defaults to False.
        continuous (bool, optional): if the SDE is continuous. Defaults to True.
        likelihood_weighting (bool, optional): if True, weight the loss with standard deviations. Defaults to False.
        eps (float, optional): parameter for sampling time. Defaults to 1e-5.

    Returns:
        Callable[[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: loss function
    """

    # Reduce operator for loss
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    # Loss function for SDEs
    def loss_fn(
        model_x: torch.nn.Module,
        model_adj: torch.nn.Module,
        model_rank2: torch.nn.Module,
        x: torch.Tensor,
        adj: torch.Tensor,
        rank2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get score functions
        score_fn_x = get_score_fn_cc(sde_x, model_x, train=train, continuous=continuous)
        score_fn_adj = get_score_fn_cc(
            sde_adj, model_adj, train=train, continuous=continuous
        )
        score_fn_rank2 = get_score_fn_cc(
            sde_rank2, model_rank2, train=train, continuous=continuous
        )

        # Sample time
        t = torch.rand(adj.shape[0], device=adj.device) * (sde_adj.T - eps) + eps
        # Get node flags from adjacency matrix
        flags = node_flags(adj)

        # Sample noise for the node features
        z_x = gen_noise(x, flags, sym=False)
        mean_x, std_x = sde_x.marginal_prob(x, t)
        # Perturb node features
        perturbed_x = mean_x + std_x[:, None, None] * z_x
        perturbed_x = mask_x(perturbed_x, flags)

        # Sample noise for the adjacency matrix
        z_adj = gen_noise(adj, flags, sym=True)
        mean_adj, std_adj = sde_adj.marginal_prob(adj, t)
        # Perturb adjacency matrix
        perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
        perturbed_adj = mask_adjs(perturbed_adj, flags)

        # Sample noise for the rank2 incidence matrix
        z_rank2 = gen_noise_rank2(rank2, adj.shape[-1], d_min, d_max, flags)
        mean_rank2, std_rank2 = sde_rank2.marginal_prob(rank2, t)
        # Perturb rank2 matrix
        perturbed_rank2 = mean_rank2 + std_rank2[:, None, None] * z_rank2
        perturbed_rank2 = mask_rank2(
            perturbed_rank2, adj.shape[-1], d_min, d_max, flags
        )

        # Compute score functions
        score_x = score_fn_x(perturbed_x, perturbed_adj, perturbed_rank2, flags, t)
        score_adj = score_fn_adj(perturbed_x, perturbed_adj, perturbed_rank2, flags, t)
        score_rank2 = score_fn_rank2(
            perturbed_x, perturbed_adj, perturbed_rank2, flags, t
        )

        # Compute losses
        if not (likelihood_weighting):
            losses_x = torch.square(score_x * std_x[:, None, None] + z_x)
            losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1)

            losses_adj = torch.square(score_adj * std_adj[:, None, None] + z_adj)
            losses_adj = reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1)

            losses_rank2 = torch.square(
                score_rank2 * std_rank2[:, None, None] + z_rank2
            )
            losses_rank2 = reduce_op(
                losses_rank2.reshape(losses_rank2.shape[0], -1), dim=-1
            )

        else:
            g2_x = sde_x.sde(torch.zeros_like(x), t)[1] ** 2
            losses_x = torch.square(score_x + z_x / std_x[:, None, None])
            losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1) * g2_x

            g2_adj = sde_adj.sde(torch.zeros_like(adj), t)[1] ** 2
            losses_adj = torch.square(score_adj + z_adj / std_adj[:, None, None])
            losses_adj = (
                reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1) * g2_adj
            )

            g2_rank2 = sde_rank2.sde(torch.zeros_like(rank2), t)[1] ** 2
            losses_rank2 = torch.square(
                score_rank2 + z_rank2 / std_rank2[:, None, None]
            )
            losses_rank2 = (
                reduce_op(losses_rank2.reshape(losses_rank2.shape[0], -1), dim=-1)
                * g2_rank2
            )

        return torch.mean(losses_x), torch.mean(losses_adj), torch.mean(losses_rank2)

    return loss_fn

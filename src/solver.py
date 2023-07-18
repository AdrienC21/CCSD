#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""solver.py: Contains the SDEs solvers, and the predictor and corrector algorithms.
The correctors consist of leveraging score-based MCMC methods.
"""

import abc
from typing import Callable, Optional, Tuple, Sequence

import torch
import numpy as np
from tqdm import trange

from src.losses import get_score_fn
from src.utils.graph_utils import mask_adjs, mask_x, gen_noise
from src.sde import VPSDE, subVPSDE, SDE


class Predictor(abc.ABC):
    """Abstract class for a predictor algorithm."""

    def __init__(
        self,
        sde: SDE,
        score_fn: Callable[
            [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
            torch.Tensor,
        ],
        probability_flow: bool = False,
    ) -> None:
        """Initialize the Predictor.

        Args:
            sde (SDE): the SDE to solve
            score_fn (Callable[ [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor, ]): the score function
            probability_flow (bool, optional): if True, use probability flow sampling. Defaults to False.
        """
        super().__init__()
        # Initialize the Predictor
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(
        self, x: torch.Tensor, t: torch.Tensor, flags: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the latent and the adjacencies.

        Args:
            x (torch.Tensor): _description_
            t (torch.Tensor): _description_
            flags (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        pass


class Corrector(abc.ABC):
    """Abstract class for a corrector algorithm."""

    def __init__(
        self,
        sde: SDE,
        score_fn: Callable[
            [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
            torch.Tensor,
        ],
        snr: float,
        scale_eps: float,
        n_steps: int,
    ) -> None:
        """Initialize the Corrector.

        Args:
            sde (SDE): the SDE to solve
            score_fn (Callable[ [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor, ]): the score function
            snr (float): signal-to-noise ratio
            scale_eps (float): scale of the noise
            n_steps (int): number of steps
        """
        super().__init__()
        # Initialize the Corrector
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.scale_eps = scale_eps
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(
        self, x: torch.Tensor, t: torch.Tensor, flags: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class EulerMaruyamaPredictor(Predictor):
    """Euler-Maruyama predictor."""

    def __init__(
        self,
        obj: str,
        sde: SDE,
        score_fn: Callable[
            [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
            torch.Tensor,
        ],
        probability_flow: bool = False,
    ) -> None:
        """Initialize the Euler-Maruyama predictor.

        Args:
            obj (str): object to update, either "x" or "adj"
            sde (SDE): the SDE to solve
            score_fn (Callable[ [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor, ]): the score function
            probability_flow (bool, optional): if True, use probability flow sampling. Defaults to False.
        """
        super().__init__(sde, score_fn, probability_flow)
        self.obj = obj

    def update_fn(
        self, x: torch.Tensor, adj: torch.Tensor, flags: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = -1.0 / self.rsde.N

        # Reverse SDE for the node features
        if self.obj == "x":
            z = gen_noise(x, flags, sym=False)
            drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=False)
            x_mean = x + drift * dt
            x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
            return x, x_mean

        # Reverse SDE for the adjacency matrix
        elif self.obj == "adj":
            z = gen_noise(adj, flags, sym=True)
            drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=True)
            adj_mean = adj + drift * dt
            adj = adj_mean + diffusion[:, None, None] * np.sqrt(-dt) * z

            return adj, adj_mean

        # Raise error if obj is not recognized
        else:
            raise NotImplementedError(
                f"Object {self.obj} not yet supported. Select from [x, adj]."
            )


class ReverseDiffusionPredictor(Predictor):
    """Reverse diffusion predictor."""

    def __init__(
        self,
        obj: str,
        sde: SDE,
        score_fn: Callable[
            [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
            torch.Tensor,
        ],
        probability_flow: bool = False,
    ):
        """Initialize the Reverse Diffusion predictor.

        Args:
            obj (str): object to update, either "x" or "adj"
            sde (SDE): the SDE to solve
            score_fn (Callable[ [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor, ]): the score function
            probability_flow (bool, optional): if True, use probability flow sampling. Defaults to False.
        """
        super().__init__(sde, score_fn, probability_flow)
        self.obj = obj

    def update_fn(
        self, x: torch.Tensor, adj: torch.Tensor, flags: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Reverse SDE for the node features
        if self.obj == "x":
            f, G = self.rsde.discretize(x, adj, flags, t, is_adj=False)
            z = gen_noise(x, flags, sym=False)
            x_mean = x - f
            x = x_mean + G[:, None, None] * z
            return x, x_mean

        # Reverse SDE for the adjacency matrix
        elif self.obj == "adj":
            f, G = self.rsde.discretize(x, adj, flags, t, is_adj=True)
            z = gen_noise(adj, flags)
            adj_mean = adj - f
            adj = adj_mean + G[:, None, None] * z
            return adj, adj_mean

        # Raise error if obj is not recognized
        else:
            raise NotImplementedError(
                f"Object {self.obj} not yet supported. Select from [x, adj]."
            )


class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(
        self,
        obj: str,
        sde: SDE,
        score_fn: Callable[
            [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
            torch.Tensor,
        ],
        snr: float,
        scale_eps: float,
        n_steps: int,
    ):
        """Initialize the NoneCorrector (an empty corrector that does nothing).

        Args:
            obj (str): object to update, either "x" or "adj"
            sde (SDE): the SDE to solve. UNUSED HERE
            score_fn (Callable[ [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor, ]): the score function. UNUSED HERE
            snr (float): signal-to-noise ratio. UNUSED HERE
            scale_eps (float): scale of the noise. UNUSED HERE
            n_steps (int): number of steps to take. UNUSED HERE
        """
        self.obj = obj
        pass

    def update_fn(
        self, x: torch.Tensor, adj: torch.Tensor, flags: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Reverse SDE for the node features
        if self.obj == "x":
            return x, x

        # Reverse SDE for the adjacency matrix
        elif self.obj == "adj":
            return adj, adj

        # Raise error if obj is not recognized
        else:
            raise NotImplementedError(
                f"Object {self.obj} not yet supported. Select from [x, adj]."
            )


class LangevinCorrector(Corrector):
    """Langevin corrector."""

    def __init__(
        self,
        obj: str,
        sde: SDE,
        score_fn: Callable[
            [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
            torch.Tensor,
        ],
        snr: float,
        scale_eps: float,
        n_steps: int,
    ):
        """Initialize the Langevin corrector.

        Args:
            obj (str): object to update, either "x" or "adj"
            sde (SDE): the SDE to solve
            score_fn (Callable[ [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor, ]): the score function
            snr (float): signal-to-noise ratio
            scale_eps (float): scale of the noise
            n_steps (int): number of steps to take
        """
        super().__init__(sde, score_fn, snr, scale_eps, n_steps)
        self.obj = obj

    def update_fn(
        self, x: torch.Tensor, adj: torch.Tensor, flags: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        seps = self.scale_eps

        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        # Reverse SDE for the node features
        if self.obj == "x":
            for _ in range(n_steps):
                grad = score_fn(x, adj, flags, t)
                noise = gen_noise(x, flags, sym=False)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(
                    noise.reshape(noise.shape[0], -1), dim=-1
                ).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                x_mean = x + step_size[:, None, None] * grad
                x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
            return x, x_mean

        # Reverse SDE for the adjacency matrix
        elif self.obj == "adj":
            for _ in range(n_steps):
                grad = score_fn(x, adj, flags, t)
                noise = gen_noise(adj, flags)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(
                    noise.reshape(noise.shape[0], -1), dim=-1
                ).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                adj_mean = adj + step_size[:, None, None] * grad
                adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
            return adj, adj_mean

        else:
            raise NotImplementedError(
                f"Object {self.obj} not yet supported. Select from [x, adj]."
            )


def get_predictor(predictor: str) -> Predictor:
    """Get the predictor function.

    Args:
        predictor (str): the predictor to use. Select from [Reverse, Euler].

    Raises:
        NotImplementedError: raise an error if the predictor is not recognized.

    Returns:
        Predictor: the predictor function.
    """
    if predictor == "Reverse":
        predictor_fn = ReverseDiffusionPredictor
    elif predictor == "Euler":
        predictor_fn = EulerMaruyamaPredictor
    else:
        raise NotImplementedError(
            f"Predictor {predictor} not yet supported. Select from [Reverse, Euler]."
        )
    return predictor_fn


def get_corrector(corrector: str) -> Corrector:
    """Get the corrector function.

    Args:
        corrector (str): the corrector to use. Select from [Langevin, None].

    Raises:
        NotImplementedError: raise an error if the corrector is not recognized.

    Returns:
        Corrector: the corrector function.
    """
    if corrector == "Langevin":
        corrector_fn = LangevinCorrector
    elif corrector == "None":
        corrector_fn = NoneCorrector
    else:
        raise NotImplementedError(
            f"Corrector {corrector} not yet supported. Select from [Langevin, None]."
        )
    return corrector_fn


def get_pc_sampler(
    sde_x: SDE,
    sde_adj: SDE,
    shape_x: Sequence[int],
    shape_adj: Sequence[int],
    predictor: str = "Euler",
    corrector: str = "None",
    snr: float = 0.1,
    scale_eps: float = 1.0,
    n_steps: int = 1,
    probability_flow: bool = False,
    continuous: bool = False,
    denoise: bool = True,
    eps: float = 1e-3,
    device: str = "cuda",
):
    """Returns a PC sampler.

    Args:
        sde_x (SDE): SDE for the node features
        sde_adj (SDE): SDE for the adjacency matrix
        shape_x (Sequence[int]): shape of the node features
        shape_adj (Sequence[int]): shape of the adjacency matrix
        predictor (str, optional): predictor function. Select from [Euler, Reverse]. Defaults to "Euler".
        corrector (str, optional): corrector function. Select from [Langevin, None]. Defaults to "None".
        snr (float, optional): signal-to-noise ratio. Defaults to 0.1.
        scale_eps (float, optional): scale of the noise. Defaults to 1.0.
        n_steps (int, optional): number of steps to take. Defaults to 1.
        probability_flow (bool, optional): if True, use probability flow sampling. Defaults to False.
        continuous (bool, optional): if True, use continuous-time SDEs, for the score function. Defaults to False.
        denoise (bool, optional): if True, use denoising diffusion (returns the mean of the reverse SDE). Defaults to True.
        eps (float, optional): epsilon for the reverse-time SDE. Defaults to 1e-3.
        device (str, optional): device to use. Defaults to "cuda".
    """

    def pc_sampler(
        model_x: torch.nn.Module, model_adj: torch.nn.Module, init_flags: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """PC sampler: sample from the model.

        Args:
            model_x (torch.nn.Module): model for the node features
            model_adj (torch.nn.Module): model for the adjacency matrix
            init_flags (torch.Tensor): initial flags

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float]: node features, adjacency matrix, timestep
        """

        # Get score functions
        score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
        score_fn_adj = get_score_fn(
            sde_adj, model_adj, train=False, continuous=continuous
        )

        # Get predictor and corrector functions
        predictor_fn = get_predictor(predictor)
        corrector_fn = get_corrector(corrector)

        # Evaluate the predictor and corrector
        predictor_obj_x = predictor_fn("x", sde_x, score_fn_x, probability_flow)
        corrector_obj_x = corrector_fn("x", sde_x, score_fn_x, snr, scale_eps, n_steps)

        predictor_obj_adj = predictor_fn("adj", sde_adj, score_fn_adj, probability_flow)
        corrector_obj_adj = corrector_fn(
            "adj", sde_adj, score_fn_adj, snr, scale_eps, n_steps
        )

        with torch.no_grad():
            # Initial sample
            x = sde_x.prior_sampling(shape_x).to(device)
            adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
            flags = init_flags
            # Mask the initial sample
            x = mask_x(x, flags)
            adj = mask_adjs(adj, flags)
            diff_steps = sde_adj.N
            timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

            # Reverse diffusion process
            for i in trange(
                0, (diff_steps), desc="[Sampling]", position=1, leave=False
            ):
                t = timesteps[i]
                vec_t = torch.ones(shape_adj[0], device=t.device) * t

                _x = x
                x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
                adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)

                _x = x
                x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
                adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
            print(" ")
            return (
                (x_mean if denoise else x),
                (adj_mean if denoise else adj),
                diff_steps * (n_steps + 1),
            )

    return pc_sampler


def S4_solver(
    sde_x: SDE,
    sde_adj: SDE,
    shape_x: Sequence[int],
    shape_adj: Sequence[int],
    predictor: str = "None",
    corrector: str = "None",
    snr: float = 0.1,
    scale_eps: float = 1.0,
    n_steps: int = 1,
    probability_flow: bool = False,
    continuous: bool = False,
    denoise: bool = True,
    eps: float = 1e-3,
    device: str = "cuda",
):
    """Returns a S4 sampler.

    Args:
        sde_x (SDE): SDE for the node features
        sde_adj (SDE): SDE for the adjacency matrix
        shape_x (Sequence[int]): shape of the node features
        shape_adj (Sequence[int]): shape of the adjacency matrix
        predictor (str, optional): predictor function. UNUSED HERE. Select from [Euler, Reverse]. Defaults to "None".
        corrector (str, optional): corrector function. UNUSED HERE. Select from [Langevin, None]. Defaults to "None".
        snr (float, optional): signal-to-noise ratio. Defaults to 0.1.
        scale_eps (float, optional): scale of the noise. Defaults to 1.0.
        n_steps (int, optional): number of steps to take. UNUSED HERE. Defaults to 1.
        probability_flow (bool, optional): if True, use probability flow sampling. UNUSED HERE. Defaults to False.
        continuous (bool, optional): if True, use continuous-time SDEs, for the score function. Defaults to False.
        denoise (bool, optional): if True, use denoising diffusion (returns the mean of the reverse SDE). Defaults to True.
        eps (float, optional): epsilon for the reverse-time SDE. Defaults to 1e-3.
        device (str, optional): device to use. Defaults to "cuda".
    """

    def s4_solver(
        model_x: torch.nn.Module, model_adj: torch.nn.Module, init_flags: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """S4 solver: sample from the model.

        Args:
            model_x (torch.nn.Module): model for the node features
            model_adj (torch.nn.Module): model for the adjacency matrix
            init_flags (torch.Tensor): initial flags

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float]: node features, adjacency matrix, timestep
        """

        # Get score functions
        score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
        score_fn_adj = get_score_fn(
            sde_adj, model_adj, train=False, continuous=continuous
        )

        with torch.no_grad():
            # Initial sample
            x = sde_x.prior_sampling(shape_x).to(device)
            adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
            flags = init_flags
            # Mask the initial sample
            x = mask_x(x, flags)
            adj = mask_adjs(adj, flags)
            diff_steps = sde_adj.N
            timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)
            dt = -1.0 / diff_steps

            # Reverse diffusion process
            for i in trange(
                0, (diff_steps), desc="[Sampling]", position=1, leave=False
            ):
                t = timesteps[i]
                vec_t = torch.ones(shape_adj[0], device=t.device) * t
                vec_dt = torch.ones(shape_adj[0], device=t.device) * (dt / 2)

                # Score computation
                score_x = score_fn_x(x, adj, flags, vec_t)
                score_adj = score_fn_adj(x, adj, flags, vec_t)

                Sdrift_x = -sde_x.sde(x, vec_t)[1][:, None, None] ** 2 * score_x
                Sdrift_adj = -sde_adj.sde(adj, vec_t)[1][:, None, None] ** 2 * score_adj

                # Correction step
                timestep = (vec_t * (sde_x.N - 1) / sde_x.T).long()

                noise = gen_noise(x, flags, sym=False)
                grad_norm = torch.norm(
                    score_x.reshape(score_x.shape[0], -1), dim=-1
                ).mean()
                noise_norm = torch.norm(
                    noise.reshape(noise.shape[0], -1), dim=-1
                ).mean()
                if isinstance(sde_x, VPSDE):
                    alpha = sde_x.alphas.to(vec_t.device)[timestep]
                else:
                    alpha = torch.ones_like(vec_t)

                step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                x_mean = x + step_size[:, None, None] * score_x
                x = (
                    x_mean
                    + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps
                )

                noise = gen_noise(adj, flags)
                grad_norm = torch.norm(
                    score_adj.reshape(score_adj.shape[0], -1), dim=-1
                ).mean()
                noise_norm = torch.norm(
                    noise.reshape(noise.shape[0], -1), dim=-1
                ).mean()
                if isinstance(sde_adj, VPSDE):
                    alpha = sde_adj.alphas.to(vec_t.device)[timestep]  # VP
                else:
                    alpha = torch.ones_like(vec_t)  # VE
                step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                adj_mean = adj + step_size[:, None, None] * score_adj
                adj = (
                    adj_mean
                    + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps
                )

                # Prediction step
                x_mean = x
                adj_mean = adj
                mu_x, sigma_x = sde_x.transition(x, vec_t, vec_dt)
                mu_adj, sigma_adj = sde_adj.transition(adj, vec_t, vec_dt)
                x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
                adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)

                x = x + Sdrift_x * dt
                adj = adj + Sdrift_adj * dt

                mu_x, sigma_x = sde_x.transition(x, vec_t + vec_dt, vec_dt)
                mu_adj, sigma_adj = sde_adj.transition(adj, vec_t + vec_dt, vec_dt)
                x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
                adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)

                x_mean = mu_x
                adj_mean = mu_adj
            print(" ")
            return (x_mean if denoise else x), (adj_mean if denoise else adj), 0

    return s4_solver

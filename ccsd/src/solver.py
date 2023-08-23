#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""solver.py: Contains the SDEs solvers, and the predictor and corrector algorithms.
The correctors consist of leveraging score-based MCMC methods.

Adapted from Jo, J. & al (2022) for Combinatorial Complexes and higher-order domain compatibility.
"""

import abc
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tqdm import trange

from ccsd.src.losses import get_score_fn, get_score_fn_cc
from ccsd.src.sde import SDE, VPSDE, subVPSDE
from ccsd.src.utils.cc_utils import gen_noise_rank2, mask_rank2
from ccsd.src.utils.graph_utils import gen_noise, mask_adjs, mask_x
from ccsd.src.utils.models_utils import get_ones


class Predictor(abc.ABC):
    """Abstract class for a predictor algorithm."""

    def __init__(
        self,
        sde: SDE,
        score_fn: Union[
            Callable[
                [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
                torch.Tensor,
            ],
            Callable[
                [
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    Optional[torch.Tensor],
                    torch.Tensor,
                ],
                torch.Tensor,
            ],
        ],
        probability_flow: bool = False,
        is_cc: bool = False,
        d_min: Optional[int] = None,
        d_max: Optional[int] = None,
    ) -> None:
        """Initialize the Predictor.

        Args:
            sde (SDE): the SDE to solve
            score_fn (Union[Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor], Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor]]): the score function
            probability_flow (bool, optional): if True, use probability flow sampling. Defaults to False.
            is_cc (bool, optional): if True, get predictor for combinatorial complexes. Defaults to False.
            d_min (Optional[int], optional): minimum size of rank-2 cells (if combinatorial complexes). Defaults to None.
            d_max (Optional[int], optional): maximum size of rank-2 cells (if combinatorial complexes). Defaults to None.
        """
        super().__init__()
        # Initialize the Predictor
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow, is_cc)
        self.score_fn = score_fn
        self.is_cc = is_cc
        self.d_min = d_min
        self.d_max = d_max

    @abc.abstractmethod
    def update_fn(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update function for the predictor class

        Args:
            x (torch.Tensor): tensor
            adj (torch.Tensor): adjacency matrix. Optional.
            rank2 (torch.Tensor): rank-2 tensor. Optional.
            flags (torch.Tensor): flags
            t (torch.Tensor): timestep

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated tensor and mean
        """
        pass


class Corrector(abc.ABC):
    """Abstract class for a corrector algorithm."""

    def __init__(
        self,
        sde: SDE,
        score_fn: Union[
            Callable[
                [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
                torch.Tensor,
            ],
            Callable[
                [
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    Optional[torch.Tensor],
                    torch.Tensor,
                ],
                torch.Tensor,
            ],
        ],
        snr: float,
        scale_eps: float,
        n_steps: int,
        is_cc: bool = False,
        d_min: Optional[int] = None,
        d_max: Optional[int] = None,
    ) -> None:
        """Initialize the Corrector.

        Args:
            sde (SDE): the SDE to solve
            score_fn (Union[Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor], Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor]]): the score function
            snr (float): signal-to-noise ratio
            scale_eps (float): scale of the noise
            n_steps (int): number of steps
            is_cc (bool, optional): if True, get corrector for combinatorial complexes. Defaults to False.
            d_min (Optional[int], optional): minimum size of rank-2 cells (if combinatorial complexes). Defaults to None.
            d_max (Optional[int], optional): maximum size of rank-2 cells (if combinatorial complexes). Defaults to None.
        """
        super().__init__()
        # Initialize the Corrector
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.scale_eps = scale_eps
        self.n_steps = n_steps
        self.is_cc = is_cc
        self.d_min = d_min
        self.d_max = d_max

    @abc.abstractmethod
    def update_fn(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update function for the corrector class.

        Args:
            x (torch.Tensor): tensor
            adj (torch.Tensor): adjacency matrix. Optional.
            rank2 (torch.Tensor): rank-2 tensor. Optional.
            flags (torch.Tensor): flags
            t (torch.Tensor): timestep

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated tensor and mean
        """
        pass


class EulerMaruyamaPredictor(Predictor):
    """Euler-Maruyama predictor."""

    def __init__(
        self,
        obj: str,
        sde: SDE,
        score_fn: Union[
            Callable[
                [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
                torch.Tensor,
            ],
            Callable[
                [
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    Optional[torch.Tensor],
                    torch.Tensor,
                ],
                torch.Tensor,
            ],
        ],
        probability_flow: bool = False,
        is_cc: bool = False,
        d_min: Optional[int] = None,
        d_max: Optional[int] = None,
    ) -> None:
        """Initialize the Euler-Maruyama predictor.

        Args:
            obj (str): object to update, either "x", "adj", or "rank2"
            sde (SDE): the SDE to solve
            score_fn (Union[Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor], Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor]]): the score function
            probability_flow (bool, optional): if True, use probability flow sampling. Defaults to False.
            is_cc (bool, optional): if True, get Euler-Maruyama predictor for combinatorial complexes. Defaults to False.
            d_min (Optional[int], optional): minimum size of rank-2 cells (if combinatorial complexes). Defaults to None.
            d_max (Optional[int], optional): maximum size of rank-2 cells (if combinatorial complexes). Defaults to None.
        """
        super().__init__(sde, score_fn, probability_flow, is_cc, d_min, d_max)
        self.obj = obj

    def __repr__(self) -> str:
        """Representation of the Euler-Maruyama predictor."""
        return f"{self.__class__.__name__}(obj={self.obj}, sde={self.sde.__class__.__name__}, probability_flow={self.probability_flow}, is_cc={self.is_cc}, d_min={self.d_min}, d_max={self.d_max})"

    def update_fn(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update function for the Euler-Maruyama predictor."""
        if self.is_cc:
            return self.update_fn_cc(*args, **kwargs)
        else:
            return self.update_fn_graph(*args, **kwargs)

    def update_fn_graph(
        self, x: torch.Tensor, adj: torch.Tensor, flags: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update function for the Euler-Maruyama predictor for graphs.

        Args:
            x (torch.Tensor): node features
            adj (torch.Tensor): adjacency matrix
            flags (torch.Tensor): flags
            t (torch.Tensor): timestep

        Raises:
            NotImplementedError: raise an error if the object to update is not recognized.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated tensor and mean
        """
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

    def update_fn_cc(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        rank2: torch.Tensor,
        flags: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update function for the Euler-Maruyama predictor for combinatorial complexes.

        Args:
            x (torch.Tensor): node features
            adj (torch.Tensor): adjacency matrix
            rank2 (torch.Tensor): rank-2 incidence matrix
            flags (torch.Tensor): flags
            t (torch.Tensor): timestep

        Raises:
            NotImplementedError: raise an error if the object to update is not recognized.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated tensor and mean
        """
        dt = -1.0 / self.rsde.N

        # Reverse SDE for the node features
        if self.obj == "x":
            z = gen_noise(x, flags, sym=False)
            drift, diffusion = self.rsde.sde(
                x, adj, rank2, flags, t, is_adj=False, is_rank2=False
            )
            x_mean = x + drift * dt
            x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
            return x, x_mean

        # Reverse SDE for the adjacency matrix
        elif self.obj == "adj":
            z = gen_noise(adj, flags, sym=True)
            drift, diffusion = self.rsde.sde(
                x, adj, rank2, flags, t, is_adj=True, is_rank2=False
            )
            adj_mean = adj + drift * dt
            adj = adj_mean + diffusion[:, None, None] * np.sqrt(-dt) * z

            return adj, adj_mean

        # Reverse SDE for the rank2 incidence matrix
        elif self.obj == "rank2":
            z = gen_noise_rank2(rank2, adj.shape[1], self.d_min, self.d_max, flags)
            drift, diffusion = self.rsde.sde(
                x, adj, rank2, flags, t, is_adj=False, is_rank2=True
            )
            rank2_mean = rank2 + drift * dt
            rank2 = rank2_mean + diffusion[:, None, None] * np.sqrt(-dt) * z

            return rank2, rank2_mean

        # Raise error if obj is not recognized
        else:
            raise NotImplementedError(
                f"Object {self.obj} not yet supported. Select from [x, adj, rank2]."
            )


class ReverseDiffusionPredictor(Predictor):
    """Reverse diffusion predictor."""

    def __init__(
        self,
        obj: str,
        sde: SDE,
        score_fn: Union[
            Callable[
                [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
                torch.Tensor,
            ],
            Callable[
                [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
                torch.Tensor,
            ],
        ],
        probability_flow: bool = False,
        is_cc: bool = False,
        d_min: Optional[int] = None,
        d_max: Optional[int] = None,
    ):
        """Initialize the Reverse Diffusion predictor.

        Args:
            obj (str): object to update, either "x", "adj" or "rank2"
            sde (SDE): the SDE to solve
            score_fn (Union[Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor], Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor]]): the score function
            probability_flow (bool, optional): if True, use probability flow sampling. Defaults to False.
            is_cc (bool, optional): if True, get Reverse Diffusion predictor for combinatorial complexes. Defaults to False.
            d_min (Optional[int], optional): minimum size of rank-2 cells (if combinatorial complexes). Defaults to None.
            d_max (Optional[int], optional): maximum size of rank-2 cells (if combinatorial complexes). Defaults to None.
        """
        super().__init__(sde, score_fn, probability_flow, is_cc, d_min, d_max)
        self.obj = obj

    def __repr__(self) -> str:
        """Representation of the Reverse Diffusion predictor."""
        return f"{self.__class__.__name__}(obj={self.obj}, sde={self.sde.__class__.__name__}, probability_flow={self.probability_flow}, is_cc={self.is_cc}, d_min={self.d_min}, d_max={self.d_max})"

    def update_fn(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update function for the Reverse Diffusion predictor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated tensor and mean
        """
        if self.is_cc:
            return self.update_fn_cc(*args, **kwargs)
        else:
            return self.update_fn_graph(*args, **kwargs)

    def update_fn_graph(
        self, x: torch.Tensor, adj: torch.Tensor, flags: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update function for the Reverse Diffusion predictor for graphs.

        Args:
            x (torch.Tensor): node features
            adj (torch.Tensor): adjacency matrix
            flags (torch.Tensor): flags
            t (torch.Tensor): timestep

        Raises:
            NotImplementedError: raise an error if the object to update is not recognized.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated tensor and mean
        """
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

    def update_fn_cc(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        rank2: torch.Tensor,
        flags: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update function for the Reverse Diffusion predictor for combinatorial complexes.

        Args:
            x (torch.Tensor): node features
            adj (torch.Tensor): adjacency matrix
            rank2 (torch.Tensor): rank-2 incidence matrix
            flags (torch.Tensor): flags
            t (torch.Tensor): timestep

        Raises:
            NotImplementedError: raise an error if the object to update is not recognized.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated tensor and mean
        """
        # Reverse SDE for the node features
        if self.obj == "x":
            f, G = self.rsde.discretize(
                x, adj, rank2, flags, t, is_adj=False, is_rank2=False
            )
            z = gen_noise(x, flags, sym=False)
            x_mean = x - f
            x = x_mean + G[:, None, None] * z
            return x, x_mean

        # Reverse SDE for the adjacency matrix
        elif self.obj == "adj":
            f, G = self.rsde.discretize(
                x, adj, rank2, flags, t, is_adj=True, is_rank2=False
            )
            z = gen_noise(adj, flags)
            adj_mean = adj - f
            adj = adj_mean + G[:, None, None] * z
            return adj, adj_mean

        # Reverse SDE for the rank2 incidence matrix
        elif self.obj == "rank2":
            f, G = self.rsde.discretize(
                x, adj, rank2, flags, t, is_adj=False, is_rank2=True
            )
            z = gen_noise_rank2(rank2, adj.shape[1], self.d_min, self.d_max, flags)
            rank2_mean = rank2 - f
            rank2 = rank2_mean + G[:, None, None] * z
            return rank2, rank2_mean

        # Raise error if obj is not recognized
        else:
            raise NotImplementedError(
                f"Object {self.obj} not yet supported. Select from [x, adj, rank2]."
            )


class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(
        self,
        obj: str,
        sde: SDE,
        score_fn: Union[
            Callable[
                [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
                torch.Tensor,
            ],
            Callable[
                [
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    Optional[torch.Tensor],
                    torch.Tensor,
                ],
                torch.Tensor,
            ],
        ],
        snr: float,
        scale_eps: float,
        n_steps: int,
        is_cc: bool = False,
        d_min: Optional[int] = None,
        d_max: Optional[int] = None,
    ):
        """Initialize the NoneCorrector (an empty corrector that does nothing).

        Args:
            obj (str): object to update, either "x" or "adj"
            sde (SDE): the SDE to solve. UNUSED HERE
            score_fn (Union[Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor], Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor]]): the score function. UNUSED HERE
            snr (float): signal-to-noise ratio. UNUSED HERE
            scale_eps (float): scale of the noise. UNUSED HERE
            n_steps (int): number of steps to take. UNUSED HERE
            is_cc (bool, optional): if True, get NoneCorrector for combinatorial complexes. Defaults to False.
            d_min (Optional[int], optional): minimum size of rank-2 cells (if combinatorial complexes). Defaults to None.
            d_max (Optional[int], optional): maximum size of rank-2 cells (if combinatorial complexes). Defaults to None.
        """
        super().__init__(sde, score_fn, snr, scale_eps, n_steps, is_cc, d_min, d_max)
        self.obj = obj

    def __repr__(self) -> str:
        """Representation of the None corrector."""
        return f"{self.__class__.__name__}(obj={self.obj}, sde={self.sde.__class__.__name__}, snr={self.snr}, scale_eps={self.scale_eps}, n_steps={self.n_steps}, is_cc={self.is_cc}, d_min={self.d_min}, d_max={self.d_max})"

    def update_fn(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update function for the NoneCorrector.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated tensor and mean
        """
        if self.is_cc:
            return self.update_fn_cc(*args, **kwargs)
        else:
            return self.update_fn_graph(*args, **kwargs)

    def update_fn_graph(
        self, x: torch.Tensor, adj: torch.Tensor, flags: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update function for the NoneCorrector for graphs.

        Args:
            x (torch.Tensor): node features
            adj (torch.Tensor): adjacency matrix
            flags (torch.Tensor): flags
            t (torch.Tensor): timestep

        Raises:
            NotImplementedError: raise an error if the object to update is not recognized.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated tensor and mean
        """
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

    def update_fn_cc(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        rank2: torch.Tensor,
        flags: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update function for the NoneCorrector for combinatorial complexes.

        Args:
            x (torch.Tensor): node features
            adj (torch.Tensor): adjacency matrix
            rank2 (torch.Tensor): rank-2 incidence matrix
            flags (torch.Tensor): flags
            t (torch.Tensor): timestep

        Raises:
            NotImplementedError: raise an error if the object to update is not recognized.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated tensor and mean
        """
        # Reverse SDE for the node features
        if self.obj == "x":
            return x, x

        # Reverse SDE for the adjacency matrix
        elif self.obj == "adj":
            return adj, adj

        # Reverse SDE for the rank2 incidence matrix
        elif self.obj == "rank2":
            return rank2, rank2

        # Raise error if obj is not recognized
        else:
            raise NotImplementedError(
                f"Object {self.obj} not yet supported. Select from [x, adj, rank2]."
            )


class LangevinCorrector(Corrector):
    """Langevin corrector."""

    def __init__(
        self,
        obj: str,
        sde: SDE,
        score_fn: Union[
            Callable[
                [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
                torch.Tensor,
            ],
            Callable[
                [
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    Optional[torch.Tensor],
                    torch.Tensor,
                ],
                torch.Tensor,
            ],
        ],
        snr: float,
        scale_eps: float,
        n_steps: int,
        is_cc: bool = False,
        d_min: Optional[int] = None,
        d_max: Optional[int] = None,
    ):
        """Initialize the Langevin corrector.

        Args:
            obj (str): object to update, either "x", "adj" or "rank2"
            sde (SDE): the SDE to solve
            score_fn (Union[Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor], Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor]]): the score function
            snr (float): signal-to-noise ratio
            scale_eps (float): scale of the noise
            n_steps (int): number of steps to take
            is_cc (bool, optional): if True, get Langevin corrector for combinatorial complexes. Defaults to False.
            d_min (Optional[int], optional): minimum size of rank-2 cells (if combinatorial complexes). Defaults to None.
            d_max (Optional[int], optional): maximum size of rank-2 cells (if combinatorial complexes). Defaults to None.
        """
        super().__init__(sde, score_fn, snr, scale_eps, n_steps, is_cc, d_min, d_max)
        self.obj = obj

    def __repr__(self) -> str:
        """Representation of the Langevin corrector."""
        return f"{self.__class__.__name__}(obj={self.obj}, sde={self.sde.__class__.__name__}, snr={self.snr}, scale_eps={self.scale_eps}, n_steps={self.n_steps}, is_cc={self.is_cc}, d_min={self.d_min}, d_max={self.d_max})"

    def update_fn(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update function for the Langevin corrector.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated tensor and mean
        """
        if self.is_cc:
            return self.update_fn_cc(*args, **kwargs)
        else:
            return self.update_fn_graph(*args, **kwargs)

    def update_fn_graph(
        self, x: torch.Tensor, adj: torch.Tensor, flags: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update function for the Langevin corrector for graphs.

        Args:
            x (torch.Tensor): node features
            adj (torch.Tensor): adjacency matrix
            flags (torch.Tensor): flags
            t (torch.Tensor): timestep

        Raises:
            NotImplementedError: raise an error if the object to update is not recognized.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated tensor and mean
        """
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        seps = self.scale_eps

        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = get_ones(t.shape, t.device)

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

    def update_fn_cc(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        rank2: torch.Tensor,
        flags: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update function for the Langevin corrector for combinatorial complexes.

        Args:
            x (torch.Tensor): node features
            adj (torch.Tensor): adjacency matrix
            rank2 (torch.Tensor): rank-2 incidence matrix
            flags (torch.Tensor): flags
            t (torch.Tensor): timestep

        Raises:
            NotImplementedError: raise an error if the object to update is not recognized.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated tensor and mean
        """
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        seps = self.scale_eps

        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = get_ones(t.shape, t.device)

        # Reverse SDE for the node features
        if self.obj == "x":
            for _ in range(n_steps):
                grad = score_fn(x, adj, rank2, flags, t)
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
                grad = score_fn(x, adj, rank2, flags, t)
                noise = gen_noise(adj, flags)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(
                    noise.reshape(noise.shape[0], -1), dim=-1
                ).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                adj_mean = adj + step_size[:, None, None] * grad
                adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
            return adj, adj_mean

        # Reverse SDE for the rank2 incidence matrix
        elif self.obj == "rank2":
            for _ in range(n_steps):
                grad = score_fn(x, adj, rank2, flags, t)
                noise = gen_noise_rank2(
                    rank2, adj.shape[1], self.d_min, self.d_max, flags
                )
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(
                    noise.reshape(noise.shape[0], -1), dim=-1
                ).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                rank2_mean = rank2 + step_size[:, None, None] * grad
                rank2 = (
                    rank2_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
                )
            return rank2, rank2_mean

        else:
            raise NotImplementedError(
                f"Object {self.obj} not yet supported. Select from [x, adj, rank2]."
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
    is_cc: bool = False,
    sde_rank2: Optional[SDE] = None,
    shape_rank2: Optional[Sequence[int]] = None,
    d_min: Optional[int] = None,
    d_max: Optional[int] = None,
) -> Union[
    Callable[
        [torch.nn.Module, torch.nn.Module, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]],
    ],
    Callable[
        [torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.Tensor],
        Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]
        ],
    ],
]:
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
        is_cc (bool, optional): if True, get PC sampler for combinatorial complexes. Defaults to False.
        sde_rank2 (Optional[SDE], optional): SDE for the higher-order features. Defaults to None.
        shape_rank2 (Optional[Sequence[int]], optional): shape of the higher-order features. Defaults to None.
        d_min (Optional[int], optional): minimum size of rank-2 cells (if combinatorial complexes). Defaults to None.
        d_max (Optional[int], optional): maximum size of rank-2 cells (if combinatorial complexes). Defaults to None.

    Returns:
        Union[Callable[[torch.nn.Module, torch.nn.Module, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]]], Callable[[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]]]]: PC sampler
    """

    if not (is_cc):

        def pc_sampler(
            model_x: torch.nn.Module,
            model_adj: torch.nn.Module,
            init_flags: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]]:
            """PC sampler: sample from the model.

            Args:
                model_x (torch.nn.Module): model for the node features
                model_adj (torch.nn.Module): model for the adjacency matrix
                init_flags (torch.Tensor): initial flags

            Returns:
                Tuple[torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]]: node features, adjacency matrix, timestep, one complete diffusion trajectory
            """

            # Get score functions
            score_fn_x = get_score_fn(
                sde_x, model_x, train=False, continuous=continuous
            )
            score_fn_adj = get_score_fn(
                sde_adj, model_adj, train=False, continuous=continuous
            )

            # Get predictor and corrector functions
            predictor_fn = get_predictor(predictor)
            corrector_fn = get_corrector(corrector)

            # Evaluate the predictor and corrector
            predictor_obj_x = predictor_fn("x", sde_x, score_fn_x, probability_flow)
            corrector_obj_x = corrector_fn(
                "x", sde_x, score_fn_x, snr, scale_eps, n_steps
            )

            predictor_obj_adj = predictor_fn(
                "adj", sde_adj, score_fn_adj, probability_flow
            )
            corrector_obj_adj = corrector_fn(
                "adj", sde_adj, score_fn_adj, snr, scale_eps, n_steps
            )

            # One complete diffusion trajectory
            diff_traj = []

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
                    vec_t = get_ones((shape_adj[0],), device=t.device) * t

                    _x = x
                    x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
                    adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)

                    _x = x
                    x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
                    adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)

                    # Add diffusion trajectory
                    if denoise:
                        diff_traj.append(
                            [x_mean[0].detach().clone(), adj_mean[0].detach().clone()]
                        )
                    else:
                        diff_traj.append(
                            [x[0].detach().clone(), adj[0].detach().clone()]
                        )

                print(" ")
                return (
                    (x_mean if denoise else x),
                    (adj_mean if denoise else adj),
                    diff_steps * (n_steps + 1),
                    diff_traj,
                )

    else:

        def pc_sampler(
            model_x: torch.nn.Module,
            model_adj: torch.nn.Module,
            model_rank2: torch.nn.Module,
            init_flags: torch.Tensor,
        ) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]
        ]:
            """PC sampler: sample from the model.

            Args:
                model_x (torch.nn.Module): model for the node features
                model_adj (torch.nn.Module): model for the adjacency matrix
                model_rank2 (torch.nn.Module): model for the higher-order features (rank2 incidence matrix)
                init_flags (torch.Tensor): initial flags

            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]]: node features, adjacency matrix, rank2 incidence matrix, timestep, one complete diffusion trajectory
            """

            # Get score functions
            score_fn_x = get_score_fn_cc(
                sde_x, model_x, train=False, continuous=continuous
            )
            score_fn_adj = get_score_fn_cc(
                sde_adj, model_adj, train=False, continuous=continuous
            )
            score_fn_rank2 = get_score_fn_cc(
                sde_rank2, model_rank2, train=False, continuous=continuous
            )

            # Get predictor and corrector functions
            predictor_fn = get_predictor(predictor)
            corrector_fn = get_corrector(corrector)

            # Evaluate the predictor and corrector
            predictor_obj_x = predictor_fn(
                "x",
                sde_x,
                score_fn_x,
                probability_flow,
                is_cc=True,
                d_min=d_min,
                d_max=d_max,
            )
            corrector_obj_x = corrector_fn(
                "x",
                sde_x,
                score_fn_x,
                snr,
                scale_eps,
                n_steps,
                is_cc=True,
                d_min=d_min,
                d_max=d_max,
            )

            predictor_obj_adj = predictor_fn(
                "adj",
                sde_adj,
                score_fn_adj,
                probability_flow,
                is_cc=True,
                d_min=d_min,
                d_max=d_max,
            )
            corrector_obj_adj = corrector_fn(
                "adj",
                sde_adj,
                score_fn_adj,
                snr,
                scale_eps,
                n_steps,
                is_cc=True,
                d_min=d_min,
                d_max=d_max,
            )

            predictor_obj_rank2 = predictor_fn(
                "rank2",
                sde_rank2,
                score_fn_rank2,
                probability_flow,
                is_cc=True,
                d_min=d_min,
                d_max=d_max,
            )
            corrector_obj_rank2 = corrector_fn(
                "rank2",
                sde_rank2,
                score_fn_rank2,
                snr,
                scale_eps,
                n_steps,
                is_cc=True,
                d_min=d_min,
                d_max=d_max,
            )

            # One complete diffusion trajectory
            diff_traj = []

            with torch.no_grad():
                # Initial sample
                x = sde_x.prior_sampling(shape_x).to(device)
                adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
                rank2 = sde_rank2.prior_sampling(shape_rank2).to(device)
                flags = init_flags
                # Mask the initial sample
                x = mask_x(x, flags)
                adj = mask_adjs(adj, flags)
                rank2 = mask_rank2(rank2, adj.shape[1], d_min, d_max, flags)
                diff_steps = sde_adj.N
                timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

                # Reverse diffusion process
                for i in trange(
                    0, (diff_steps), desc="[Sampling]", position=1, leave=False
                ):
                    t = timesteps[i]
                    vec_t = get_ones((shape_adj[0],), device=t.device) * t

                    _x = x
                    _adj = adj
                    x, x_mean = corrector_obj_x.update_fn(x, adj, rank2, flags, vec_t)
                    adj, adj_mean = corrector_obj_adj.update_fn(
                        _x, adj, rank2, flags, vec_t
                    )
                    rank2, rank2_mean = corrector_obj_rank2.update_fn(
                        _x, _adj, rank2, flags, vec_t
                    )

                    _x = x
                    _adj = adj
                    x, x_mean = predictor_obj_x.update_fn(x, adj, rank2, flags, vec_t)
                    adj, adj_mean = predictor_obj_adj.update_fn(
                        _x, adj, rank2, flags, vec_t
                    )
                    rank2, rank2_mean = predictor_obj_rank2.update_fn(
                        _x, _adj, rank2, flags, vec_t
                    )

                    # Add diffusion trajectory
                    if denoise:
                        diff_traj.append(
                            [
                                x_mean[0].detach().clone(),
                                adj_mean[0].detach().clone(),
                                rank2_mean[0].detach().clone(),
                            ]
                        )
                    else:
                        diff_traj.append(
                            [
                                x[0].detach().clone(),
                                adj[0].detach().clone(),
                                rank2[0].detach().clone(),
                            ]
                        )

                print(" ")
                return (
                    (x_mean if denoise else x),
                    (adj_mean if denoise else adj),
                    (rank2_mean if denoise else rank2),
                    diff_steps * (n_steps + 1),
                    diff_traj,
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
    is_cc: bool = False,
    sde_rank2: Optional[SDE] = None,
    shape_rank2: Optional[Sequence[int]] = None,
    d_min: Optional[int] = None,
    d_max: Optional[int] = None,
) -> Union[
    Callable[
        [torch.nn.Module, torch.nn.Module, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]],
    ],
    Callable[
        [torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.Tensor],
        Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]
        ],
    ],
]:
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
        is_cc (bool, optional): if True, get S4 sampler for combinatorial complexes. Defaults to False.
        sde_rank2 (Optional[SDE], optional): SDE for the higher-order features. Defaults to None.
        shape_rank2 (Optional[Sequence[int]], optional): shape of the higher-order features. Defaults to None.
        d_min (Optional[int], optional): minimum size of rank-2 cells (if combinatorial complexes). Defaults to None.
        d_max (Optional[int], optional): maximum size of rank-2 cells (if combinatorial complexes). Defaults to None.

    Returns:
        Union[Callable[[torch.nn.Module, torch.nn.Module, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]]], Callable[[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]]]]: S4 sampler
    """

    if not (is_cc):

        def s4_solver(
            model_x: torch.nn.Module,
            model_adj: torch.nn.Module,
            init_flags: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]]:
            """S4 solver: sample from the model.

            Args:
                model_x (torch.nn.Module): model for the node features
                model_adj (torch.nn.Module): model for the adjacency matrix
                init_flags (torch.Tensor): initial flags

            Returns:
                Tuple[torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]]: node features, adjacency matrix, timestep, one complete diffusion trajectory
            """

            # Get score functions
            score_fn_x = get_score_fn(
                sde_x, model_x, train=False, continuous=continuous
            )
            score_fn_adj = get_score_fn(
                sde_adj, model_adj, train=False, continuous=continuous
            )

            # One complete diffusion trajectory
            diff_traj = []

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
                    vec_t = get_ones((shape_adj[0],), device=t.device) * t
                    vec_dt = get_ones((shape_adj[0],), device=t.device) * (dt / 2)

                    # Score computation
                    score_x = score_fn_x(x, adj, flags, vec_t)
                    score_adj = score_fn_adj(x, adj, flags, vec_t)

                    Sdrift_x = -sde_x.sde(x, vec_t)[1][:, None, None] ** 2 * score_x
                    Sdrift_adj = (
                        -sde_adj.sde(adj, vec_t)[1][:, None, None] ** 2 * score_adj
                    )

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
                        alpha = get_ones(vec_t.shape, vec_t.device)

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
                        alpha = get_ones(vec_t.shape, vec_t.device)  # VE
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

                    # Add diffusion trajectory
                    if denoise:
                        diff_traj.append(
                            [x_mean[0].detach().clone(), adj_mean[0].detach().clone()]
                        )
                    else:
                        diff_traj.append(
                            [x[0].detach().clone(), adj[0].detach().clone()]
                        )

                print(" ")
                return (
                    (x_mean if denoise else x),
                    (adj_mean if denoise else adj),
                    0,
                    diff_traj,
                )

    else:

        def s4_solver(
            model_x: torch.nn.Module,
            model_adj: torch.nn.Module,
            model_rank2: torch.nn.Module,
            init_flags: torch.Tensor,
        ) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]
        ]:
            """S4 solver: sample from the model.

            Args:
                model_x (torch.nn.Module): model for the node features
                model_adj (torch.nn.Module): model for the adjacency matrix
                model_rank2 (torch.nn.Module): model for the higher-order features (rank2 incidence matrix)
                init_flags (torch.Tensor): initial flags

            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]]]: node features, adjacency matrix, incidence matrix, timestep, one complete diffusion trajectory
            """

            # Get score functions
            score_fn_x = get_score_fn_cc(
                sde_x, model_x, train=False, continuous=continuous
            )
            score_fn_adj = get_score_fn_cc(
                sde_adj, model_adj, train=False, continuous=continuous
            )
            score_fn_rank2 = get_score_fn_cc(
                sde_rank2, model_rank2, train=False, continuous=continuous
            )

            # One complete diffusion trajectory
            diff_traj = []

            with torch.no_grad():
                # Initial sample
                x = sde_x.prior_sampling(shape_x).to(device)
                adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
                rank2 = sde_rank2.prior_sampling(shape_rank2).to(device)
                flags = init_flags
                # Mask the initial sample
                x = mask_x(x, flags)
                adj = mask_adjs(adj, flags)
                rank2 = mask_rank2(rank2, adj.shape[1], d_min, d_max, flags)
                diff_steps = sde_adj.N
                timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)
                dt = -1.0 / diff_steps

                # Reverse diffusion process
                for i in trange(
                    0, (diff_steps), desc="[Sampling]", position=1, leave=False
                ):
                    t = timesteps[i]
                    vec_t = get_ones((shape_adj[0],), device=t.device) * t
                    vec_dt = get_ones((shape_adj[0],), device=t.device) * (dt / 2)

                    # Score computation
                    score_x = score_fn_x(x, adj, rank2, flags, vec_t)
                    score_adj = score_fn_adj(x, adj, rank2, flags, vec_t)
                    score_rank2 = score_fn_rank2(x, adj, rank2, flags, vec_t)

                    Sdrift_x = -sde_x.sde(x, vec_t)[1][:, None, None] ** 2 * score_x
                    Sdrift_adj = (
                        -sde_adj.sde(adj, vec_t)[1][:, None, None] ** 2 * score_adj
                    )
                    Sdrift_rank2 = (
                        -sde_rank2.sde(rank2, vec_t)[1][:, None, None] ** 2
                        * score_rank2
                    )

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
                        alpha = get_ones(vec_t.shape, vec_t.device)

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
                        alpha = get_ones(vec_t.shape, vec_t.device)  # VE
                    step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                    adj_mean = adj + step_size[:, None, None] * score_adj
                    adj = (
                        adj_mean
                        + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps
                    )

                    noise = gen_noise_rank2(rank2, adj.shape[1], d_min, d_max, flags)
                    grad_norm = torch.norm(
                        score_rank2.reshape(score_rank2.shape[0], -1), dim=-1
                    ).mean()
                    noise_norm = torch.norm(
                        noise.reshape(noise.shape[0], -1), dim=-1
                    ).mean()
                    if isinstance(sde_rank2, VPSDE):
                        alpha = sde_rank2.alphas.to(vec_t.device)[timestep]
                    else:
                        alpha = get_ones(vec_t.shape, vec_t.device)

                    step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                    rank2_mean = rank2 + step_size[:, None, None] * score_rank2
                    rank2 = (
                        rank2_mean
                        + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps
                    )

                    # Prediction step
                    x_mean = x
                    adj_mean = adj
                    rank2_mean = rank2
                    mu_x, sigma_x = sde_x.transition(x, vec_t, vec_dt)
                    mu_adj, sigma_adj = sde_adj.transition(adj, vec_t, vec_dt)
                    mu_rank2, sigma_rank2 = sde_rank2.transition(rank2, vec_t, vec_dt)
                    x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
                    adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)
                    rank2 = mu_rank2 + sigma_rank2[:, None, None] * gen_noise_rank2(
                        rank2, adj.shape[1], d_min, d_max, flags
                    )

                    x = x + Sdrift_x * dt
                    adj = adj + Sdrift_adj * dt
                    rank2 = rank2 + Sdrift_rank2 * dt

                    mu_x, sigma_x = sde_x.transition(x, vec_t + vec_dt, vec_dt)
                    mu_adj, sigma_adj = sde_adj.transition(adj, vec_t + vec_dt, vec_dt)
                    mu_rank2, sigma_rank2 = sde_rank2.transition(
                        rank2, vec_t + vec_dt, vec_dt
                    )
                    x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
                    adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)
                    rank2 = mu_rank2 + sigma_rank2[:, None, None] * gen_noise_rank2(
                        rank2, adj.shape[1], d_min, d_max, flags
                    )

                    x_mean = mu_x
                    adj_mean = mu_adj
                    rank2_mean = mu_rank2

                    # Add diffusion trajectory
                    if denoise:
                        diff_traj.append(
                            [
                                x_mean[0].detach().clone(),
                                adj_mean[0].detach().clone(),
                                rank2_mean[0].detach().clone(),
                            ]
                        )
                    else:
                        diff_traj.append(
                            [
                                x[0].detach().clone(),
                                adj[0].detach().clone(),
                                rank2[0].detach().clone(),
                            ]
                        )

                print(" ")
                return (
                    (x_mean if denoise else x),
                    (adj_mean if denoise else adj),
                    (rank2_mean if denoise else rank2),
                    0,
                    diff_traj,
                )

    return s4_solver

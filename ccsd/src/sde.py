#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""sde.py: contains the different Stochastic Differential Equations (SDEs) classes: VPSDE, VESDE, subVPSDE.
The classes inherit from the SDE class.

Adapted from Jo, J. & al (2022)
"""

import abc
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch


class SDE(abc.ABC):
    """SDE abstract class. All functions are designed for a mini-batch of inputs."""

    def __init__(self, N: int) -> None:
        """Initialize a SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self) -> int:
        """Return the final time of the SDE.

        Returns:
            int: final time of the SDE.
        """
        pass

    @abc.abstractmethod
    def sde(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parameters to determine the drift and diffusion functions of the SDE, $f_t(x)$ and $G_t(x)$.

        Args:
            x (torch.Tensor): feature vector.
            t (torch.Tensor): time step (from 0 to `self.T`).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: drift and diffusion.
        """
        pass

    @abc.abstractmethod
    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$.

        Args:
            x (torch.Tensor): feature vector.
            t (torch.Tensor): time step (from 0 to `self.T`).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: mean and standard deviation of the perturbation kernel.
        """
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape: Sequence[int]) -> torch.Tensor:
        """Generate one sample from the prior distribution, $p_T(x)$.

        Args:
            shape (Sequence[int]): shape of the sample.

        Returns:
            torch.Tensor: sample from the prior distribution.
        """
        pass

    @abc.abstractmethod
    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log-density of the prior distribution.
        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z (torch.Tensor): latent sample.
        Returns:
            torch.Tensor: log probability density
        """
        pass

    def discretize(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x (torch.Tensor): torch tensor
            t (torch.Tensor): torch float representing the time step (from 0 to `self.T`)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: drift and diffusion (f, G).
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(
        self,
        score_fn: Callable[
            [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor],
            torch.Tensor,
        ],
        probability_flow: bool = False,
        is_cc: bool = False,
    ) -> "SDE":
        """Create the reverse-time SDE/ODE (RSDE).

        Args:
            score_fn (Union[Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor], Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], torch.Tensor]]): time-dependent score-based model that takes x and t and returns the score.
            probability_flow (bool, optional): If `True`, create the reverse-time ODE used for probability flow sampling. Defaults to False.
            is_cc (bool, optional): If `True`, create the reverse-time SDE/ODE takes the rank2 incidence matrix as an input. Defaults to False.

        Returns:
            SDE: reverse-time SDE/ODE.
        """

        N = self.N
        T = self.T
        self.is_cc = is_cc
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE
        if not (is_cc):

            class RSDE(self.__class__):
                """Reverse-time SDE/ODE."""

                def __init__(self) -> None:
                    """Initialize the reverse-time SDE/ODE."""
                    self.N = N
                    self.probability_flow = probability_flow
                    self.is_cc = is_cc

                def __repr__(self) -> str:
                    """Return the string representation of the reverse-time SDE/ODE.

                    Returns:
                        str: string representation of the reverse-time SDE/ODE.
                    """
                    return f"{self.__class__.__name__}(N={self.N}, probability_flow={probability_flow}, T={self.T})"

                @property
                def T(self) -> int:
                    """Return the final time of the reverse-time SDE/ODE.

                    Returns:
                        int: final time of the reverse-time SDE/ODE.
                    """
                    return T

                def sde(
                    self,
                    feature: torch.Tensor,
                    x: torch.Tensor,
                    flags: torch.Tensor,
                    t: torch.Tensor,
                    is_adj: bool = True,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    """Returns the drift and diffusion for the reverse SDE/ODE.

                    Args:
                        feature (torch.Tensor): torch tensor.
                        x (torch.Tensor): torch tensor.
                        flags (torch.Tensor): flags
                        t (torch.Tensor): torch float representing the time step (from 0 to `self.T`)
                        is_adj (bool, optional): True if reverse-SDE for the adjacency matrix. Defaults to True.

                    Returns:
                        Tuple[torch.Tensor, torch.Tensor]: drift and diffusion.
                    """
                    drift, diffusion = sde_fn(x, t) if is_adj else sde_fn(feature, t)
                    score = score_fn(feature, x, flags, t)
                    drift = drift - diffusion[:, None, None] ** 2 * score * (
                        0.5 if self.probability_flow else 1.0
                    )
                    # Set the diffusion function to zero for ODEs.
                    diffusion = 0.0 if self.probability_flow else diffusion
                    return drift, diffusion

                def discretize(
                    self,
                    feature: torch.Tensor,
                    x: torch.Tensor,
                    flags: torch.Tensor,
                    t: torch.Tensor,
                    is_adj: bool = True,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    """Create discretized iteration rules for the reverse diffusion sampler.

                    Args:
                        feature (torch.Tensor): torch tensor.
                        x (torch.Tensor): torch tensor.
                        flags (torch.Tensor): flags
                        t (torch.Tensor): torch float representing the time step (from 0 to `self.T`)
                        is_adj (bool, optional): True if reverse-SDE for the adjacency matrix. Defaults to True.

                    Returns:
                        Tuple[torch.Tensor, torch.Tensor]: discretized drift and diffusion (f, G).
                    """
                    f, G = discretize_fn(x, t) if is_adj else discretize_fn(feature, t)
                    score = score_fn(feature, x, flags, t)
                    rev_f = f - G[:, None, None] ** 2 * score * (
                        0.5 if self.probability_flow else 1.0
                    )
                    rev_G = torch.zeros_like(G) if self.probability_flow else G
                    return rev_f, rev_G

        else:

            class RSDE(self.__class__):
                """Reverse-time SDE/ODE."""

                def __init__(self) -> None:
                    """Initialize the reverse-time SDE/ODE."""
                    self.N = N
                    self.probability_flow = probability_flow
                    self.is_cc = is_cc

                def __repr__(self) -> str:
                    """Return the string representation of the reverse-time SDE/ODE.

                    Returns:
                        str: string representation of the reverse-time SDE/ODE.
                    """
                    return f"{self.__class__.__name__}(N={self.N}, probability_flow={probability_flow}, T={self.T})"

                @property
                def T(self) -> int:
                    """Return the final time of the reverse-time SDE/ODE.

                    Returns:
                        int: final time of the reverse-time SDE/ODE.
                    """
                    return T

                def sde(
                    self,
                    feature: torch.Tensor,
                    x: torch.Tensor,
                    rank2: torch.Tensor,
                    flags: torch.Tensor,
                    t: torch.Tensor,
                    is_adj: bool = True,
                    is_rank2: bool = False,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    """Returns the drift and diffusion for the reverse SDE/ODE.

                    Args:
                        feature (torch.Tensor): torch tensor.
                        x (torch.Tensor): torch tensor.
                        rank2 (torch.Tensor): torch tensor.
                        flags (torch.Tensor): flags
                        t (torch.Tensor): torch float representing the time step (from 0 to `self.T`)
                        is_adj (bool, optional): True if reverse-SDE for the adjacency matrix. Defaults to True.
                        is_rank2 (bool, optional): True if reverse-SDE for the rank2 incidence matrix. Defaults to False.
                            is_adj needs to be set on False too.

                    Returns:
                        Tuple[torch.Tensor, torch.Tensor]: drift and diffusion.
                    """
                    if is_adj:
                        drift, diffusion = sde_fn(x, t)
                    elif is_rank2:
                        drift, diffusion = sde_fn(rank2, t)
                    else:
                        drift, diffusion = sde_fn(feature, t)
                    score = score_fn(feature, x, rank2, flags, t)
                    drift = drift - diffusion[:, None, None] ** 2 * score * (
                        0.5 if self.probability_flow else 1.0
                    )
                    # Set the diffusion function to zero for ODEs.
                    diffusion = 0.0 if self.probability_flow else diffusion
                    return drift, diffusion

                def discretize(
                    self,
                    feature: torch.Tensor,
                    x: torch.Tensor,
                    rank2: torch.Tensor,
                    flags: torch.Tensor,
                    t: torch.Tensor,
                    is_adj: bool = True,
                    is_rank2: bool = False,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    """Create discretized iteration rules for the reverse diffusion sampler.

                    Args:
                        feature (torch.Tensor): torch tensor.
                        x (torch.Tensor): torch tensor.
                        rank2 (torch.Tensor): torch tensor.
                        flags (torch.Tensor): flags
                        t (torch.Tensor): torch float representing the time step (from 0 to `self.T`)
                        is_adj (bool, optional): True if reverse-SDE for the adjacency matrix. Defaults to True.
                        is_rank2 (bool, optional): True if reverse-SDE for the rank2 incidence matrix. Defaults to False.
                            is_adj needs to be set on False too.

                    Returns:
                        Tuple[torch.Tensor, torch.Tensor]: discretized drift and diffusion (f, G).
                    """
                    if is_adj:
                        f, G = discretize_fn(x, t)
                    elif is_rank2:
                        f, G = discretize_fn(rank2, t)
                    else:
                        f, G = discretize_fn(feature, t)
                    score = score_fn(feature, x, rank2, flags, t)
                    rev_f = f - G[:, None, None] ** 2 * score * (
                        0.5 if self.probability_flow else 1.0
                    )
                    rev_G = torch.zeros_like(G) if self.probability_flow else G
                    return rev_f, rev_G

        return RSDE()


class VPSDE(SDE):
    """Variance Preserving SDE (VPSDE)."""

    def __init__(
        self, beta_min: float = 0.1, beta_max: float = 20.0, N: int = 1000
    ) -> None:
        """Construct a Variance Preserving SDE.

        Args:
            beta_min (float): value of beta(0)
            beta_max (float): value of beta(1)
            N (int): number of discretization steps
        """

        super().__init__(N)
        # Initialize the parameters
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def __repr__(self) -> str:
        """Return the string representation of the SDE.

        Returns:
            str: string representation of the SDE.
        """
        return f"{self.__class__.__name__}(N={self.N}, beta_min={self.beta_0}, beta_max={self.beta_1})"

    @property
    def T(self) -> int:
        """Return the final time of the SDE.

        Returns:
            int: final time of the SDE.
        """
        return 1

    def sde(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the drift and diffusion for the SDE.

        Args:
            x (torch.Tensor): torch tensor.
            t (torch.Tensor): torch float representing the time step (from 0 to `self.T`)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: drift and diffusion.
        """

        # beta(t) = beta_min + t * (beta_max - beta_min)
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the mean and std of the perturbation kernel.

        Args:
            x (torch.Tensor): torch tensor.
            t (torch.Tensor): torch float representing the time step (from 0 to `self.T`)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: mean and std of the perturbation kernel.
        """

        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape: Sequence[int]) -> torch.Tensor:
        """Sample from the prior distribution.
        Here the prior is a standard Gaussian distribution.

        Args:
            shape (Sequence[int]): shape of the output tensor.

        Returns:
            torch.Tensor: sample from the prior distribution.
        """
        return torch.randn(*shape)

    def prior_sampling_sym(self, shape: Sequence[int]) -> torch.Tensor:
        """Sample from the prior distribution in the symmetric case for a matrix.
        Here the prior is a standard Gaussian distribution.

        Args:
            shape (Sequence[int]): shape of the output tensor.

        Returns:
            torch.Tensor: sample from the prior distribution.
        """
        x = torch.randn(*shape).triu(1)
        return x + x.transpose(-1, -2)

    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        """Returns the log probability of the prior distribution.

        Args:
            z (torch.Tensor): latent sample.

        Returns:
            torch.Tensor: log probability of the prior distribution.
        """
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2)) / 2.0
        return logps

    def discretize(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DDPM discretization for the drift and diffusion of the SDE.

        Args:
            x (torch.Tensor): torch tensor.
            t (torch.Tensor): torch float representing the time step (from 0 to `self.T`)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: discretized drift and diffusion (f, G).
        """
        timestep = (t * (self.N - 1) / self.T).long().to("cpu")
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None] * x - x
        G = sqrt_beta
        return f, G

    def transition(
        self, x: torch.Tensor, t: torch.Tensor, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the mean and std of the transition kernel.

        Args:
            x (torch.Tensor): torch tensor.
            t (torch.Tensor): torch float representing the time step (from 0 to `self.T`)
            dt (float): time step (here negative timestep dt).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: mean and std of the transition kernel.
        """
        log_mean_coeff = (
            0.25 * dt * (2 * self.beta_0 + (2 * t + dt) * (self.beta_1 - self.beta_0))
        )
        mean = torch.exp(-log_mean_coeff[:, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std


class VESDE(SDE):
    """Variance Exploding SDE (VESDE)."""

    def __init__(
        self, sigma_min: float = 0.01, sigma_max: float = 50.0, N: int = 1000
    ) -> None:
        """Initialize the Variance Exploding SDE.

        Args:
            sigma_min (float): smallest sigma.
            sigma_max (float): largest sigma.
            N (int): number of discretization steps
        """
        super().__init__(N)
        # Initialize the parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N)
        )
        self.N = N

    def __repr__(self) -> str:
        """Return the string representation of the SDE.

        Returns:
            str: string representation of the SDE.
        """
        return f"{self.__class__.__name__}(N={self.N}, beta_min={self.beta_0}, beta_max={self.beta_1})"

    @property
    def T(self) -> int:
        """Return the final time of the SDE.

        Returns:
            int: final time of the SDE.
        """
        return 1

    def sde(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the drift and diffusion of the SDE.

        Args:
            x (torch.Tensor): torch tensor.
            t (torch.Tensor): torch float representing the time step (from 0 to `self.T`)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: drift and diffusion of the SDE.
        """
        # sigma(t) = sigma_min * (sigma_max / sigma_min) ** t
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(
            torch.tensor(
                2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device
            )
        )
        return drift, diffusion

    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the mean and std of the marginal distribution at time t.

        Args:
            x (torch.Tensor): torch tensor.
            t (torch.Tensor): torch float representing the time step (from 0 to `self.T`)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: mean and std of the marginal distribution.
        """
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape: Sequence[int]) -> torch.Tensor:
        """Returns a sample from the prior distribution.
        Here the prior is a standard Gaussian distribution.

        Args:
            shape (Sequence[int]): shape of the sample.

        Returns:
            torch.Tensor: sample from the prior distribution.
        """
        return torch.randn(*shape)

    def prior_sampling_sym(self, shape: Sequence[int]) -> torch.Tensor:
        """Returns a sample from the prior distribution.
        Here the prior is a standard Gaussian distribution.
        Symmetric version of the prior sampling.

        Args:
            shape (Sequence[int]): shape of the sample.

        Returns:
            torch.Tensor: sample from the prior distribution.
        """
        x = torch.randn(*shape).triu(1)
        x = x + x.transpose(-1, -2)
        return x

    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        """Returns the log probability of the prior distribution.

        Args:
            z (torch.Tensor): latent sample.

        Returns:
            torch.Tensor: log probability of the prior distribution.
        """
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(
            z**2, dim=(1, 2, 3)
        ) / (2 * self.sigma_max**2)

    def discretize(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the drift and diffusion of the discretized SDE.
        SMLD(NCSN) discretization

        Args:
            x (torch.Tensor): torch tensor.
            t (torch.Tensor): torch float representing the time step (from 0 to `self.T`)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: drift and diffusion of the discretized SDE.
        """

        timestep = (t * (self.N - 1) / self.T).long().to("cpu")
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(
            timestep.to(t.device) == 0,
            torch.zeros_like(t),
            self.discrete_sigmas[timestep - 1].to(t.device),
        )
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma**2 - adjacent_sigma**2)
        return f, G

    def transition(
        self, x: torch.Tensor, t: torch.Tensor, dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the mean and std of the transition kernel at time t and timestep dt.
        (negative timestep dt, means going backward in time)

        Args:
            x (torch.Tensor): torch tensor.
            t (torch.Tensor): torch float representing the time step (from 0 to `self.T`)
            dt (float): timestep

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: mean and std of the transition kernel.
        """
        std = torch.square(
            self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        ) - torch.square(self.sigma_min * (self.sigma_max / self.sigma_min) ** (t + dt))
        std = torch.sqrt(std)
        mean = x
        return mean, std


class subVPSDE(SDE):
    """Class for the sub-VP SDE that excels at likelihoods."""

    def __init__(
        self, beta_min: float = 0.1, beta_max: float = 20.0, N: int = 1000
    ) -> None:
        """Construct the sub-VP SDE that excels at likelihoods.
        Args:
            beta_min (float): value of beta(0)
            beta_max (float): value of beta(1)
            N (int): number of discretization steps
        """
        super().__init__(N)
        # Initialize the parameters
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas

    def __repr__(self) -> str:
        """Return the string representation of the SDE.

        Returns:
            str: string representation of the SDE.
        """
        return f"{self.__class__.__name__}(N={self.N}, beta_min={self.beta_0}, beta_max={self.beta_1})"

    @property
    def T(self) -> int:
        """Returns the final time of the SDE.

        Returns:
            int: final time of the SDE.
        """
        return 1

    def sde(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the drift and diffusion of the SDE at time t.

        Args:
            x (torch.Tensor): torch tensor.
            t (torch.Tensor): torch float representing the time step (from 0 to `self.T`)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: drift and diffusion of the SDE.
        """
        # beta(t) = beta_min + t * (beta_max - beta_min)
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None] * x
        discount = 1.0 - torch.exp(
            -2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2
        )
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the mean and std of the marginal distribution at time t.

        Args:
            x (torch.Tensor): torch tensor.
            t (torch.Tensor): torch float representing the time step (from 0 to `self.T`)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: mean and std of the marginal distribution.
        """
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff)[:, None, None] * x
        std = 1 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape: Sequence[int]) -> torch.Tensor:
        """Returns a sample from the prior distribution.
        Here, the prior distribution is a standard Gaussian.

        Args:
            shape (Sequence[int]): shape of the sample.

        Returns:
            torch.Tensor: sample from the prior distribution.
        """
        return torch.randn(*shape)

    def prior_sampling_sym(self, shape: Sequence[int]) -> torch.Tensor:
        """Returns a sample from the prior distribution.
        Here, the prior distribution is a standard Gaussian.
        Symmetric version of the prior sampling.

        Args:
            shape (Sequence[int]): shape of the sample.

        Returns:
            torch.Tensor: sample from the prior distribution.
        """
        x = torch.randn(*shape).triu(1)
        return x + x.transpose(-1, -2)

    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        """Returns the log probability of the prior distribution.

        Args:
            z (torch.Tensor): latent sample.

        Returns:
            torch.Tensor: log probability of the prior distribution.
        """
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0

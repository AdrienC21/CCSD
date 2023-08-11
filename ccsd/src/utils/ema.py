#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ema.py: code for the exponential moving average class for the parameters.

Adapted from Jo, J. & al (2022), almost left untouched.
"""

from typing import Any, Dict

import torch


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """

    def __init__(
        self,
        parameters: torch.nn.parameter.Parameter,
        decay: float,
        use_num_updates: bool = True,
    ) -> None:
        """Initialize the EMA class.

        Args:
            parameters (torch.nn.parameter.Parameter): Iterable of `torch.nn.Parameter`, initial
                parameters to use for EMA.
            decay (float): Decay rate for exponential moving average.
            use_num_updates (bool, optional): if True, initialize the number of updates to 0. Defaults to True.

        Raises:
            ValueError: raise an error if decay is not between 0 and 1.
        """

        if (decay < 0.0) or (decay > 1.0):
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def __repr__(self) -> str:
        """Return the string representation of the EMA class.

        Returns:
            str: the string representation of the EMA class
        """
        return f"ExponentialMovingAverage(decay={self.decay}, num_updates={self.num_updates}, shadow_params={self.shadow_params}, collected_params={self.collected_params})"

    def update(self, parameters: torch.nn.parameter.Parameter) -> None:
        """Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters (torch.nn.parameter.Parameter): Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters: torch.nn.parameter.Parameter) -> None:
        """Copy current parameters into given collection of parameters.

        Args:
            parameters (torch.nn.parameter.Parameter): Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters: torch.nn.parameter.Parameter) -> None:
        """Save the current parameters for restoring later.

        Args:
            parameters (torch.nn.parameter.Parameter): Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters: torch.nn.parameter.Parameter) -> None:
        """Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters (torch.nn.parameter.Parameter): Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary containing the state of the EMA.

        Returns:
            Dict[str, Any]: dictionary containing the state of the EMA.
        """
        return dict(
            decay=self.decay,
            num_updates=self.num_updates,
            shadow_params=self.shadow_params,
        )

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the dictionary containing the state of the EMA.

        Args:
            state_dict (Dict[str, Any]): _description_
        """
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = state_dict["shadow_params"]

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""run_frobenius_complexity.py: Code to assess the complexity of learning partial score functions using the Frobenius norm of the Jacobian of our models.
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import torch
import torch.autograd as autograd

from ccsd.src.parsers.config import get_config
from ccsd.src.utils.loader import load_model_optimizer, load_model_params
from ccsd.src.utils.models_utils import get_nb_parameters


def frobenius_norm_jacobian(model: torch.nn.Module, t: torch.Tensor) -> float:
    """Calculate the Frobenius norm of the Jacobian matrix of a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model for which to compute the Jacobian.
        t (torch.Tensor): Input tensor for which to compute the Jacobian.

    Returns:
        float: The Frobenius norm of the Jacobian matrix of the model for the input tensor.
    """
    # Evaluation mode and clear gradients
    model.eval()
    model.zero_grad()
    # Calculate the Jacobian matrix
    jac = autograd.functional.jacobian(model, t)
    # Compute the Frobenius norm
    frob_norm = torch.norm(jac, "fro").item()
    return frob_norm


if __name__ == "__main__":
    for dataset in [
        "QM9",
        "ENZYMES_small",
        "community_small",
        "ego_small",
        "grid_small",
    ]:
        t = None  # TO PROVIDE

        print("\n----------------------")
        print(f"{dataset}")
        print("-----")

        print("\nGraph")
        cfg = f"{dataset.lower()}"
        config = get_config(cfg, 42)
        params_x, params_adj = load_model_params(config, is_cc=False)
        try:
            model_x, optimizer_x, scheduler_x = load_model_optimizer(
                params_x, config.train, "cpu"
            )
            model_adj, optimizer_adj, scheduler_adj = load_model_optimizer(
                params_adj, config.train, "cpu"
            )

            print(f"Complexity x: {frobenius_norm_jacobian(model_x, t)}")
            print(f"Complexity adj: {frobenius_norm_jacobian(model_adj, t)}")
        except Exception as e:
            print("NaN")

        print("\nCC")
        cfg = f"{dataset.lower()}_CC"
        config = get_config(cfg, 42)
        params_x, params_adj, params_rank2 = load_model_params(config, is_cc=True)
        try:
            model_x, optimizer_x, scheduler_x = load_model_optimizer(
                params_x, config.train, "cpu"
            )
            model_adj, optimizer_adj, scheduler_adj = load_model_optimizer(
                params_adj, config.train, "cpu"
            )
            model_rank2, optimizer_rank2, scheduler_rank2 = load_model_optimizer(
                params_rank2, config.train, "cpu"
            )

            print(f"Complexity x: {frobenius_norm_jacobian(model_x, t)}")
            print(f"Complexity adj: {frobenius_norm_jacobian(model_adj, t)}")
            print(f"Complexity rank2: {frobenius_norm_jacobian(model_rank2, t)}")
        except Exception as e:
            print("NaN")

        print("\nCC Base Ablation study")
        cfg = f"{dataset.lower()}_Base_CC"
        config = get_config(cfg, 42)
        params_x, params_adj, params_rank2 = load_model_params(config, is_cc=True)
        try:
            model_x, optimizer_x, scheduler_x = load_model_optimizer(
                params_x, config.train, "cpu"
            )
            model_adj, optimizer_adj, scheduler_adj = load_model_optimizer(
                params_adj, config.train, "cpu"
            )
            model_rank2, optimizer_rank2, scheduler_rank2 = load_model_optimizer(
                params_rank2, config.train, "cpu"
            )

            print(f"Complexity x: {frobenius_norm_jacobian(model_x, t)}")
            print(f"Complexity adj: {frobenius_norm_jacobian(model_adj, t)}")
            print(f"Complexity rank2: {frobenius_norm_jacobian(model_rank2, t)}")
        except Exception as e:
            print("NaN")

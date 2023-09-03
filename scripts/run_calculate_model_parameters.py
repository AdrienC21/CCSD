#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""run_calculate_model_parameters.py: Code to compare the number of parameters of the models for the different datasets in function of some configurations.
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import torch

from ccsd.src.parsers.config import get_config
from ccsd.src.utils.loader import load_model_optimizer, load_model_params
from ccsd.src.utils.models_utils import get_nb_parameters

if __name__ == "__main__":
    for dataset in [
        "QM9",
        "ENZYMES_small",
        "community_small",
        "ego_small",
        "grid_small",
    ]:
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

            print(f"X: {get_nb_parameters(model_x)}")
            print(f"A: {get_nb_parameters(model_adj)}")
            tot_param_graph = sum(
                [get_nb_parameters(model) for model in [model_x, model_adj]]
            )
            print(f"Total: {tot_param_graph}")
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

            print(f"X: {get_nb_parameters(model_x)}")
            print(f"A: {get_nb_parameters(model_adj)}")
            print(f"F: {get_nb_parameters(model_rank2)}")
            tot_param_cc = sum(
                [
                    get_nb_parameters(model)
                    for model in [model_x, model_adj, model_rank2]
                ]
            )
            print(f"Total: {tot_param_cc}")
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

            print(f"X: {get_nb_parameters(model_x)}")
            print(f"A: {get_nb_parameters(model_adj)}")
            print(f"F: {get_nb_parameters(model_rank2)}")
            tot_param_cc_base = sum(
                [
                    get_nb_parameters(model)
                    for model in [model_x, model_adj, model_rank2]
                ]
            )
            print(f"Total: {tot_param_cc_base}")
        except Exception as e:
            print("NaN")

        min_val = min([tot_param_graph, tot_param_cc, tot_param_cc_base])
        max_val = max([tot_param_graph, tot_param_cc, tot_param_cc_base])
        print(f"Max MAPE diff: {100 * (max_val - min_val) / max_val}")

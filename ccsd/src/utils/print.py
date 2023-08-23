#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""print.py: utility functions for printing to the console.
"""

import argparse
import os
from typing import Any, Dict, Union


def get_ascii_logo(ascii_logo_path: str = "ascii_logo.txt") -> str:
    """Get the ascii logo.

    Args:
        ascii_logo_path (str, optional): path of the logo. Defaults to "ascii_logo.txt".

    Returns:
        str: the ascii logo.
    """
    if not (os.path.exists(ascii_logo_path)):
        ascii_logo_path = os.path.join(os.path.dirname(__file__), ascii_logo_path)
        if not (os.path.exists(ascii_logo_path)):
            print("No ascii logo found.")
            return ""

    with open(ascii_logo_path, "r") as f:
        ascii_logo = f.read()

    return ascii_logo


def get_experiment_desc(args: Union[argparse.Namespace, Dict[str, Any]]) -> str:
    """Get the experiment description.

    Args:
        args (Union[argparse.Namespace, Dict[str, Any]]): parsed arguments for the experiment.

    Returns:
        str: the experiment description.
    """

    experiment_desc = "Current experiment:\n\n"
    if isinstance(args, argparse.Namespace):
        for arg in vars(args):
            experiment_desc += f"\t{arg}: {getattr(args, arg)}\n"
    else:
        for arg in args:
            experiment_desc += f"\t{arg}: {args[arg]}\n"

    return experiment_desc


def initial_print(
    args: Union[argparse.Namespace, Dict[str, Any]],
    ascii_logo_path: str = "ascii_logo.txt",
) -> None:
    """Print the initial message to the console.

    Args:
        args (Union[argparse.Namespace, Dict[str, Any]]): parsed arguments for the experiment.
        ascii_logo_path (str, optional): path of the logo. Defaults to "ascii_logo.txt".
    """

    # Get the ascii logo and the experiment description
    ascii_logo = get_ascii_logo(os.path.join(args.folder, ascii_logo_path))
    experiment_desc = get_experiment_desc(args)

    # Print the initial message
    print(ascii_logo)
    print(100 * "-")
    print(experiment_desc)

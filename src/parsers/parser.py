#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""parser.py: code for parsing the arguments of the main script (experiments).

Adapted from Jo, J. & al (2022)

Almost left untouched.
"""

import argparse


class Parser:
    """Parser class to parse the arguments to run the experiments."""

    def __init__(self) -> None:
        """Initialize the parser."""

        self.parser = argparse.ArgumentParser(
            description="CCSD - Combinatorial Complex Stochastic Diffusion"
        )
        self.set_arguments()

    def set_arguments(self) -> None:
        """Set the arguments for the parser."""
        self.parser.add_argument(
            "--type",
            type=str,
            required=True,
            choices=["train", "sample"],
            help="Type of experiment",
        )
        self.parser.add_argument(
            "--config", type=str, required=True, help="Path of config file"
        )
        self.parser.add_argument(
            "--folder",
            type=str,
            default="./",
            help="Directory to save the results, load checkpoints, load config, etc",
        )
        self.parser.add_argument(
            "--comment",
            type=str,
            default="",
            help="A single line comment for the experiment",
        )
        self.parser.add_argument(
            "--seed", type=int, default=42, help="Random seed for reproducibility"
        )

    def parse(self) -> argparse.Namespace:
        """Parse the arguments and check for unknown arguments.

        Raises:
            SystemExit: raise an error if there are unknown arguments.

        Returns:
            argparse.Namespace: parsed arguments.
        """
        args, unparsed = self.parser.parse_known_args()

        if len(unparsed) != 0:  # print if there are unknown arguments
            raise SystemExit("Unknown argument(s): {}".format(unparsed))

        return args

    def __repr__(self) -> str:
        """Return the string representation of the parser."""
        return self.__class__.__name__

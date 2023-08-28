#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""parser_generator.py: code for parsing the arguments of the graph dataset generator.
"""

import argparse


class ParserGenerator:
    """ParserGenerator class to parse the arguments to create graph datasets."""

    def __init__(self) -> None:
        """Initialize the parser."""

        self.parser = argparse.ArgumentParser(description="Graph dataset generator")
        self.set_arguments()

    def set_arguments(self) -> None:
        """Set the arguments for the parser."""
        self.parser.add_argument(
            "--data-dir",
            type=str,
            default="data",
            help="directory to save the generated dataset",
        )
        self.parser.add_argument(
            "--dataset",
            type=str,
            default="community_small",
            help="type of dataset to generate",
            choices=[
                "ego_small",
                "community_small",
                "ENZYMES",
                "ENZYMES_small",
                "grid",
            ],
        )
        self.parser.add_argument(
            "--is_cc",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="if you want to generate combinatorial complexes instead of graphs",
        )
        self.parser.add_argument(
            "--folder",
            type=str,
            default="./",
            help="Directory to save the results, load checkpoints, load config, etc",
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

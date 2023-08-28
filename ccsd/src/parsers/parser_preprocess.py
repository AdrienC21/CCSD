#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""parser_preprocess.py: code for parsing the arguments of the scripts
that preprocess the molecule datasets.
"""

import argparse


class ParserPreprocess:
    """ParserPreprocess class to parse the arguments of the scripts
    that preprocess the molecule datasets."""

    def __init__(self) -> None:
        """Initialize the parser."""

        self.parser = argparse.ArgumentParser(
            description="Preprocess the test molecules"
        )
        self.set_arguments()

    def set_arguments(self) -> None:
        """Set the arguments for the parser."""
        self.parser.add_argument(
            "--dataset",
            type=str,
            default="QM9",
            choices=["QM9", "ZINC250k"],
            help="Dataset name",
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

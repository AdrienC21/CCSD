#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""preprocess.py: preprocess the molecule datasets (not for NSPDK).

Adapted from Jo, J. & al (2022)
"""

import argparse
import os
import sys
import time
from time import perf_counter

sys.path.insert(0, os.getcwd())

import pandas as pd

from ccsd.data.utils.data_frame_parser import DataFrameParser
from ccsd.data.utils.numpytupledataset import NumpyTupleDataset
from ccsd.data.utils.smile_to_graph import GGNNPreprocessor
from ccsd.src.parsers.parser_preprocess import ParserPreprocess


def preprocess(args: argparse.Namespace, print_elapsed_time: bool = True) -> None:
    """Preprocess the molecules (not for NSPDK)

    Adapted from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow

    Args:
        args (argparse.Namespace): arguments
        print_elapsed_time (bool, optional): if True, print the elapsed time to preprocess the molecules. Defaults to True.

    Raises:
        ValueError: raise an error if the dataset is not supported.
            Molecule dataset supported: QM9, ZINC250k
    """
    start_time = perf_counter()
    data_name = args.dataset
    folder = args.folder

    if data_name == "ZINC250k":
        max_atoms = 38
        path = os.path.join(folder, "data", "zinc250k.csv")
        smiles_col = "smiles"
        label_idx = 1
    elif data_name == "QM9":
        max_atoms = 9
        path = os.path.join(folder, "data", "qm9.csv")
        smiles_col = "SMILES1"
        label_idx = 2
    else:
        raise ValueError(
            f"[ERROR] Unexpected value. Dataset {data_name} not supported."
        )

    preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)

    print(f"Preprocessing {data_name} data")
    df = pd.read_csv(path, index_col=0)
    # Caution: Not reasonable but used in chain_chemistry\datasets\zinc.py:
    # "smiles" column contains '\n', need to remove it.
    # Here we do not remove \n, because it represents atom N with single bond
    labels = df.keys().tolist()[label_idx:]
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col=smiles_col)
    result = parser.parse(df, return_smiles=True)

    dataset = result["dataset"]
    smiles = result["smiles"]

    NumpyTupleDataset.save(
        os.path.join(folder, "data", f"{data_name.lower()}_kekulized.npz"), dataset
    )

    if print_elapsed_time:
        print(
            "Total time:",
            time.strftime("%H:%M:%S", time.gmtime(perf_counter() - start_time)),
        )


if __name__ == "__main__":
    # Parse the arguments
    args = ParserPreprocess().parse()
    # Preprocess the molecules
    preprocess(args)

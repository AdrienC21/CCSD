#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""preprocess_for_nspdk.py: preprocess the test molecules for NSPDK.
"""

from time import time
import os
import argparse
import sys

sys.path.insert(0, os.getcwd())

import pickle
import json
import pandas as pd

from utils.mol_utils import mols_to_nx, smiles_to_mols
from parsers.parser_preprocess import ParserPreprocess


def preprocess_nspdk(args: argparse.Namespace, print_elapsed_time: bool = True) -> None:
    """Preprocess the test molecules for NSPDK

    Args:
        args (argparse.Namespace): arguments
        print_elapsed_time (bool, True): if True, print the elapsed time to preprocess the test molecules.
            Defaults to True.

    Raises:
        ValueError: raise an error if the dataset is not supported.
            Molecule dataset supported: QM9, ZINC250k
    """

    dataset = args.dataset
    start_time = time()

    # Load the test indices
    with open(f"data/valid_idx_{dataset.lower()}.json") as f:
        test_idx = json.load(f)

    # Get the column name of the SMILES
    if dataset == "QM9":  # special case for QM9
        test_idx = test_idx["valid_idxs"]
        test_idx = [int(i) for i in test_idx]
        col = "SMILES1"
    elif dataset == "ZINC250k":
        col = "smiles"
    else:
        raise ValueError(f"[ERROR] Unexpected value. Dataset {dataset} not supported.")

    # Load the molecules
    smiles = pd.read_csv(f"data/{dataset.lower()}.csv")[col]
    # Get the test molecules
    test_smiles = [smiles.iloc[i] for i in test_idx]
    # Convert the test molecules into graphs
    nx_graphs = mols_to_nx(smiles_to_mols(test_smiles))
    print(f"Converted the test molecules into {len(nx_graphs)} graphs")

    # Save the graphs
    with open(f"data/{dataset.lower()}_test_nx.pkl", "wb") as f:
        pickle.dump(nx_graphs, f)

    # Print the elapsed time
    if print_elapsed_time:
        print(f"Total {time() - start_time:.2f} sec elapsed")


if __name__ == "__main__":
    # Parse the arguments
    args = ParserPreprocess().parse()
    # Preprocess the test molecules
    preprocess_nspdk(args)

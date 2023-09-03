#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""run_calculate_D.py: Code to calculate d_min and d_max for a combinatorial complex dataset.
Example:
    $ python scripts/run_calculate_D.py --file data/grid_small_CC.pkl
"""

import argparse
import math
import os
import pickle
import sys
from typing import List, Tuple

sys.path.insert(0, os.getcwd())

from toponetx.classes.combinatorial_complex import CombinatorialComplex
from tqdm import tqdm


def calculate_D(ccs: List[CombinatorialComplex]) -> Tuple[int, int]:
    """Calculate d_min and d_max for a combinatorial complex dataset.

    Args:
        ccs (List[CombinatorialComplex]): list of combinatorial complexes

    Returns:
        Tuple[int, int]: d_min and d_max
    """
    d_min = sys.maxsize
    for cc_idx in tqdm(list(range(len(ccs)))):
        cc = ccs[cc_idx]
        d_min = min(
            d_min,
            min(
                (len(cell) for cell in cc.cells.hyperedge_dict.get(2, {})),
                default=sys.maxsize,
            ),
        )

    d_max = -sys.maxsize
    for cc_idx in tqdm(list(range(len(ccs)))):
        cc = ccs[cc_idx]
        d_max = max(
            d_max,
            max(
                (len(cell) for cell in cc.cells.hyperedge_dict.get(2, {})),
                default=-sys.maxsize,
            ),
        )
    return d_min, d_max


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate D (d_min and d_max) for a combinatorial complex dataset"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the file containing the combinatorial complex dataset, starting from `folder`",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="./",
        help="Directory to where the path to the combinatorial complex dataset is",
    )
    args = parser.parse_known_args()[0]
    with open(os.path.join(args.folder, args.file), "rb") as f:
        ccs = pickle.load(f)
    d_min, d_max = calculate_D(ccs)
    print(f"d_min: {d_min}, d_max: {d_max}")

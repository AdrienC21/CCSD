#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""run_tanimoto.py: Code to find the most similar molecules in a training set to a set of generated molecules using Tanimoto Similarity.
"""

import argparse
import math
import os
import pickle
import sys
from typing import List, Tuple, Union

sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from tqdm import tqdm

from ccsd.src.utils.mol_utils import canonicalize_smiles, load_smiles


def find_max_similarity_molecules_tanimoto(
    generated_molecules: List[Chem.Mol],
    training_molecules: List[Chem.Mol],
    plot_result: bool = True,
    folder: str = "./",
    max_num: int = 16,
    dataset: str = "QM9",
) -> Tuple[List[Chem.Mol], float]:
    """Find the most similar molecules in a training set to a set of generated molecules using Tanimoto Similarity.

    Args:
        generated_molecules (List[Chem.Mol]): list of generated molecules
        training_molecules (List[Chem.Mol]): list of training molecules
        plot_result (bool, optional): whether to plot the most similar molecules. Defaults to True.
        folder (str, optional): directory where to create a analysis folder to save the results. Defaults to "./".
        max_num (int, optional): maximum number of molecules to plot, if we plot. Defaults to 16.
        dataset (str, optional): dataset used for the analysis. Defaults to "QM9".

    Returns:
        Tuple[List[Chem.Mol], float]: list of most similar molecules in the training set and the maximum tanimoto similarity score
    """
    # Calculate Morgan fingerprints for training molecules
    if isinstance(training_molecules):
        training_molecules = [Chem.MolFromSmiles(training_molecules)]
    training_fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        for mol in training_molecules
    ]

    max_similarity = 0.0
    max_similar_molecules = []

    print("Finding most similar molecules...")
    for mol_idx in tqdm(range(len(generated_molecules))):
        gen_mol = generated_molecules[mol_idx]
        # Calculate Morgan fingerprint for the generated molecule
        gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2, nBits=1024)

        # Calculate Tanimoto similarity with all training molecules
        similarities = BulkTanimotoSimilarity(gen_fp, training_fps)

        # Find the maximum similarity and the corresponding training molecule
        max_sim = max(similarities)
        if max_sim > max_similarity:
            max_similarity = max_sim
            max_similar_molecules = [
                training_molecules[i]
                for i, sim in enumerate(similarities)
                if sim == max_sim
            ]

    if plot_result:
        print("Plotting results...")
        if not (os.path.exists(os.path.join(folder, "analysis"))):
            os.makedirs(os.path.join(folder, "analysis"))
        max_num = min(len(max_similar_molecules), max_num)
        img_c = int(math.ceil(np.sqrt(max_num)))
        figure = plt.figure()

        for i in range(max_num):
            mol = max_similar_molecules[i]

            assert isinstance(
                mol, Chem.Mol
            ), "elements should be molecules"  # check if we have a molecule

            ax = plt.subplot(img_c, img_c, i + 1)
            mol_img = Draw.MolToImage(mol, size=(300, 300))
            ax.imshow(mol_img)
            title_str = f"{Chem.MolToSmiles(mol)}"
            ax.title.set_text(title_str)
            ax.set_axis_off()
        figure.suptitle(f"Dataset: {dataset}. Tanimoto Similarity: {max_similarity}")
        plt.savefig(
            os.path.join(
                folder, "analysis", f"{dataset}_most_similar_molecules_tanimoto.png"
            )
        )

    return max_similar_molecules, max_similarity


if __name__ == "__main__":
    PLOT_RESULT = True
    MAX_NUM = 16

    parser = argparse.ArgumentParser(description="Tanimoto analysis")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["QM9", "ZINC250k"],
        help="Dataset to use for the analysis",
    )
    parser.add_argument(
        "--single_mol",
        type=str,
        help="SMILES string of a single molecule to use for the analysis",
    )
    parser.add_argument(
        "--gen_mol_file",
        type=str,
        help="Path to the file containing the generated molecules, starting from `folder`",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="./",
        help="Directory to save the results in an `analysis` folder",
    )
    args = parser.parse_known_args()[0]

    if args.single_mol is not None:
        print("Loading single molecule...")
        train_smiles = args.single_mol
        train_molecules = [Chem.MolFromSmiles(m) for m in [train_smiles]]
    else:
        print("Loading train data...")
        train_smiles, _ = load_smiles(args.dataset)
        train_smiles = canonicalize_smiles(train_smiles)
        train_molecules = [Chem.MolFromSmiles(m) for m in train_smiles]
    print("Loading gen data...")
    with open(os.path.join(args.folder, args.gen_mol_file), "rb") as f:
        gen_molecules = pickle.load(f)
    find_max_similarity_molecules_tanimoto(
        gen_molecules, train_molecules, PLOT_RESULT, args.folder, MAX_NUM, args.dataset
    )

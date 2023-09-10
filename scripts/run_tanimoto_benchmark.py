#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""run_tanimoto_benchmark.py: Code to compare the average Tanimoto similarity between samples drawn from GDSS/CCSD and the training set.
"""

import os
import pickle
import sys

sys.path.insert(0, os.getcwd())

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from tqdm import tqdm

from ccsd.src.utils.mol_utils import canonicalize_smiles, load_smiles

if __name__ == "__main__":
    # Calculate Morgan fingerprints for QM9 training molecules
    print("Loading train data...")
    train_smiles, _ = load_smiles("QM9")
    train_smiles = canonicalize_smiles(train_smiles)
    train_molecules = [Chem.MolFromSmiles(m) for m in train_smiles]
    training_fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        for mol in train_molecules
    ]

    # CCSD
    # Load generated molecules
    gen_mol_file = (
        "samples/pkl/QM9/test/sample_qm9_CC_ccsd_qm9_CC-sample_Aug27-10-44-56_mols.pkl"
    )
    with open(os.path.join("./", gen_mol_file), "rb") as f:
        gen_molecules = pickle.load(f)

    # Calculate average Tanimoto similarity
    avg_sim = 0
    for mol_idx in tqdm(range(len(gen_molecules))):
        gen_mol = gen_molecules[mol_idx]
        # Calculate Morgan fingerprint for the generated molecule
        gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2, nBits=1024)

        # Calculate Tanimoto similarity with all training molecules
        similarities = BulkTanimotoSimilarity(gen_fp, training_fps)
        avg_sim += max(similarities)
    avg_sim /= len(gen_molecules)
    print(f"CCSD: {round(avg_sim, 3)}")

    # GDSS
    # Load generated molecules
    gen_mol_file = "samples/pkl/QM9/test/sample_qm9_retrained_gdss_qm9_retrained-sample_Aug27-14-01-35_mols.pkl"
    with open(os.path.join("./", gen_mol_file), "rb") as f:
        gen_molecules = pickle.load(f)

    # Calculate average Tanimoto similarity
    avg_sim2 = 0
    for mol_idx in tqdm(range(len(gen_mol_file))):
        gen_mol = gen_mol_file[mol_idx]
        # Calculate Morgan fingerprint for the generated molecule
        gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2, nBits=1024)

        # Calculate Tanimoto similarity with all training molecules
        similarities = BulkTanimotoSimilarity(gen_fp, training_fps)
        avg_sim2 += max(similarities)
    avg_sim2 /= len(gen_mol_file)

    print(f"GDSS: {round(avg_sim2, 3)}")

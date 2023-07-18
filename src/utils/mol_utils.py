#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""mol_utils.py: utility functions for loading the molecular data, checking the validity of the molecules, converting them, saving them, etc.
"""

from typing import List, Tuple, Optional
import re

import torch
import json
import numpy as np
import pandas as pd
import networkx as nx
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")

ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
bond_decoder = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
}
AN_TO_SYMBOL = {
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


def mols_to_smiles(mols: List[Chem.Mol]) -> List[str]:
    """Converts a list of RDKit molecules to a list of SMILES strings.

    Args:
        mols (List[Chem.Mol]): molecules to convert

    Returns:
        List[str]: SMILES strings
    """
    return [Chem.MolToSmiles(mol) for mol in mols]


def smiles_to_mols(smiles: List[str]) -> List[Chem.Mol]:
    """Converts a list of SMILES strings to a list of RDKit molecules.

    Args:
        smiles (List[str]): SMILES strings to convert

    Returns:
        List[Chem.Mol]: molecules
    """
    return [Chem.MolFromSmiles(s) for s in smiles]


def canonicalize_smiles(smiles: List[str]) -> List[str]:
    """Canonicalizes a list of SMILES strings.

    Args:
        smiles (List[str]): SMILES strings to canonicalize

    Returns:
        List[str]: canonicalized SMILES strings
    """
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]


def load_smiles(dataset: str = "QM9") -> Tuple[List[str], List[str]]:
    """Loads SMILES strings from a dataset and return train and test splits.

    Args:
        dataset (str, optional): smiles dataset to load. Defaults to "QM9".

    Raises:
        ValueError: raise an error if dataset is not supported

    Returns:
        Tuple[List[str], List[str]]: train and test splits
    """
    if dataset == "QM9":
        col = "SMILES1"
    else:
        raise ValueError(f"Wrong dataset name {dataset} in load_smiles")

    df = pd.read_csv(f"data/{dataset.lower()}.csv")

    with open(f"data/valid_idx_{dataset.lower()}.json") as f:
        test_idx = json.load(f)

    if dataset == "QM9":  # special case for QM9
        test_idx = test_idx["valid_idxs"]
        test_idx = [int(i) for i in test_idx]

    train_idx = [i for i in range(len(df)) if i not in test_idx]

    return list(df[col].loc[train_idx]), list(df[col].loc[test_idx])


def gen_mol(
    x: torch.Tensor,
    adj: torch.Tensor,
    dataset: str,
    largest_connected_comp: bool = True,
) -> Tuple[List[Chem.Mol], int]:
    """Generates molecules from the model output and returns valid molecules and the number of molecules that are not corrected.

    Args:
        x (torch.Tensor): node features
        adj (torch.Tensor): adjacency matrix
        dataset (str): dataset name
        largest_connected_comp (bool, optional): whether or not we keep only the largest connected component. Defaults to True.

    Returns:
        Tuple[List[Chem.Mol], int]: valid molecules and the number of molecules that are not corrected
    """
    # x: 32, 9, 5; adj: 32, 4, 9, 9
    x = x.detach().cpu().numpy()
    adj = adj.detach().cpu().numpy()

    if dataset == "QM9":
        atomic_num_list = [6, 7, 8, 9, 0]
    else:
        atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
    # mols_wo_correction = [valid_mol_can_with_seg(construct_mol(x_elem, adj_elem, atomic_num_list)) for x_elem, adj_elem in zip(x, adj)]
    # mols_wo_correction = [mol for mol in mols_wo_correction if mol is not None]
    mols, num_no_correct = [], 0
    for x_elem, adj_elem in zip(x, adj):
        mol = construct_mol(x_elem, adj_elem, atomic_num_list)
        cmol, no_correct = correct_mol(mol)
        if no_correct:
            num_no_correct += 1
        vcmol = valid_mol_can_with_seg(
            cmol, largest_connected_comp=largest_connected_comp
        )
        mols.append(vcmol)
    mols = [mol for mol in mols if mol is not None]
    return mols, num_no_correct


def construct_mol(
    x: torch.Tensor, adj: torch.Tensor, atomic_num_list: List[int]
) -> Chem.Mol:
    """Constructs a molecule from the model output.

    Args:
        x (torch.Tensor): node features
        adj (torch.Tensor): adjacency matrix
        atomic_num_list (List[int]): atomic number list

    Returns:
        Chem.Mol: molecule
    """
    # x: 9, 5; adj: 4, 9, 9
    mol = Chem.RWMol()

    atoms = np.argmax(x, axis=1)
    atoms_exist = atoms != len(atomic_num_list) - 1
    atoms = atoms[atoms_exist]  # 9,
    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    adj = np.argmax(adj, axis=0)  # 9, 9
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1  # bonds 0, 1, 2, 3 -> 1, 2, 3, 0 (0 denotes the virtual bond)

    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder[adj[start, end]])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def check_valency(mol: Chem.Mol) -> Tuple[bool, Optional[List[int]]]:
    """Checks the valency of the molecule.

    Args:
        mol (Chem.Mol): molecule

    Returns:
        Tuple[bool, Optional[List[int]]]: whether or not the molecule is valid and the atom id and valency of the atom that is not valid
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def correct_mol(m: Chem.Mol) -> Tuple[Chem.Mol, bool]:
    """Corrects the molecule.

    Args:
        m (Chem.Mol): molecule

    Returns:
        Tuple[Chem.Mol, bool]: corrected molecule and whether or not the molecule is corrected
    """

    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)  # in valid_mol_can_with_seg
    mol = m  # memory issue

    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (
                        b.GetIdx(),
                        int(b.GetBondType()),
                        b.GetBeginAtomIdx(),
                        b.GetEndAtomIdx(),
                    )
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder[t])
    return mol, no_correct


def valid_mol_can_with_seg(
    m: Optional[Chem.Mol], largest_connected_comp: bool = True
) -> Optional[Chem.Mol]:
    """Returns a valid molecule with the largest connected component (in option).

    Args:
        m (Optional[Chem.Mol]): molecule
        largest_connected_comp (bool, optional): whether or not we keep only the largest connected component. Defaults to True.

    Returns:
        Optional[Chem.Mol]: valid molecule
    """
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and ("." in sm):
        vsm = [
            (s, len(s)) for s in sm.split(".")
        ]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


def mols_to_nx(mols: List[Chem.Mol]) -> List[nx.Graph]:
    """Converts a list of molecules to a list of networkx graphs.

    Args:
        mols (List[Chem.Mol]): list of molecules

    Returns:
        List[nx.Graph]: list of networkx graphs
    """
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), label=atom.GetSymbol())
            # Other potential features:
            # atomic_num=atom.GetAtomicNum()
            # formal_charge=atom.GetFormalCharge()
            # chiral_tag=atom.GetChiralTag()
            # hybridization=atom.GetHybridization()
            # num_explicit_hs=atom.GetNumExplicitHs()
            # is_aromatic=atom.GetIsAromatic()

        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                label=int(bond.GetBondTypeAsDouble()),
            )
            # Other potential feature:
            # bond_type=bond.GetBondType()

        nx_graphs.append(G)
    return nx_graphs

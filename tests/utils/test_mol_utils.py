#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_mol_utils.py: test functions for mol_utils.py
"""

from typing import List

import networkx as nx
import numpy as np
import pytest
import torch
from easydict import EasyDict
from rdkit import Chem

from ccsd.src.utils.mol_utils import (
    canonicalize_smiles,
    check_valency,
    construct_mol,
    correct_mol,
    gen_mol,
    is_molecular_config,
    load_smiles,
    mols_to_nx,
    mols_to_smiles,
    smiles_to_mols,
    valid_mol_can_with_seg,
)


def test_is_molecular_config() -> None:
    """Test is_molecular_config function"""
    config_qm9 = EasyDict(data=EasyDict(data="QM9"))
    config_non_qm9 = EasyDict(data=EasyDict(data="NotQM9"))

    assert is_molecular_config(config_qm9) == True
    assert is_molecular_config(config_non_qm9) == False


def test_mols_to_smiles() -> None:
    """Test mols_to_smiles function"""
    mol1 = Chem.MolFromSmiles("CCO")
    mol2 = Chem.MolFromSmiles("C=O")
    mol3 = Chem.MolFromSmiles("CC#N")
    mols = [mol1, mol2, mol3]

    expected_smiles = ["CCO", "C=O", "CC#N"]
    assert mols_to_smiles(mols) == expected_smiles


def test_smiles_to_mols() -> None:
    """Test smiles_to_mols function"""
    smiles = ["CCO", "C=O", "CC#N"]
    expected_mols = [
        Chem.MolFromSmiles("CCO"),
        Chem.MolFromSmiles("C=O"),
        Chem.MolFromSmiles("CC#N"),
    ]

    assert mols_to_smiles(smiles_to_mols(smiles)) == mols_to_smiles(expected_mols)


def test_canonicalize_smiles() -> None:
    """Test canonicalize_smiles function"""
    smiles = ["CCO", "C=O", "CC#N"]
    canonicalized_smiles = ["CCO", "C=O", "CC#N"]

    assert canonicalize_smiles(smiles) == canonicalized_smiles


def test_load_smiles() -> None:
    """Test load_smiles function when an invalid dataset name is given"""
    with pytest.raises(ValueError, match=r"Wrong dataset name .* in load_smiles"):
        load_smiles("InvalidDatasetName")


def test_construct_mol() -> None:
    """Test construct_mol function"""
    x = np.array([[0.4, 0.3, 0.1, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0, 0.0]])
    adj = np.array(
        [
            [[0, 0.1], [0.1, 0]],
            [[0, 0.7], [0.7, 0]],
            [[0, 0.1], [0.1, 0]],
            [[0, 0.1], [0.1, 0]],
        ]
    )
    atomic_num_list = [6, 7, 8, 9, 0]

    mol = construct_mol(x, adj, atomic_num_list)

    # Make sure the constructed molecule is valid
    assert isinstance(mol, Chem.Mol)
    assert mol.GetNumAtoms() == 2
    assert mol.GetNumBonds() == 1
    bond = list(mol.GetBonds())[0]
    assert bond.GetBondTypeAsDouble() == 2.0  # double bond


def test_gen_mol() -> None:
    """Test gen_mol function"""
    x = torch.tensor([[[0.4, 0.3, 0.1, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0, 0.0]]])
    adj = torch.tensor(
        [
            [
                [[0, 0.1], [0.1, 0]],
                [[0, 0.7], [0.7, 0]],
                [[0, 0.1], [0.1, 0]],
                [[0, 0.1], [0.1, 0]],
            ]
        ]
    )
    dataset = "QM9"

    mols, num_no_correct = gen_mol(x, adj, dataset)

    assert len(mols) == 1
    assert num_no_correct == 0
    assert isinstance(mols[0], Chem.Mol)


def test_check_valency_valid_molecule() -> None:
    """Test check_valency function with a valid molecule"""
    mol = smiles_to_mols(["CCO"])[0]
    result, error_atoms = check_valency(mol)
    assert result is True
    assert error_atoms is None


def test_check_valency_invalid_molecule() -> None:
    """Test check_valency function with an invalid molecule"""
    mw = Chem.RWMol()
    mw.AddAtom(Chem.Atom(6))
    mw.AddAtom(Chem.Atom(6))
    mw.AddBond(0, 1, Chem.BondType.QUINTUPLE)
    mol = mw.GetMol()
    result, error_atoms = check_valency(mol)
    assert result is False
    assert error_atoms == [0, 5]


def test_correct_mol_valid_molecule() -> None:
    """Test correct_mol function with a valid molecule"""
    mol = smiles_to_mols(["CCO"])[0]
    _, no_correct = correct_mol(Chem.RWMol(mol))
    assert no_correct is True


def test_correct_mol_invalid_molecule() -> None:
    """Test correct_mol function with an invalid molecule"""
    mw = Chem.RWMol()
    mw.AddAtom(Chem.Atom(6))
    mw.AddAtom(Chem.Atom(6))
    mw.AddBond(0, 1, Chem.BondType.QUINTUPLE)
    mol = mw.GetMol()
    corrected_mol, no_correct = correct_mol(Chem.RWMol(mol))
    assert no_correct is False
    assert mols_to_smiles([corrected_mol])[0] == "C$C"


def test_valid_mol_can_with_seg_single_molecule() -> None:
    """Test valid_mol_can_with_seg function with a single molecule"""
    mol = smiles_to_mols(["CCO"])[0]
    valid_mol = valid_mol_can_with_seg(mol)
    assert valid_mol is not None
    assert mols_to_smiles([valid_mol])[0] == "CCO"


def test_valid_mol_can_with_seg_multiple_molecules() -> None:
    """Test valid_mol_can_with_seg function with multiple molecules"""
    mol = smiles_to_mols(["C1CC1.C2CC2"])[0]
    valid_mol = valid_mol_can_with_seg(mol)
    assert valid_mol is not None
    assert mols_to_smiles([valid_mol])[0] == "C1CC1"


def test_mols_to_nx_single_molecule() -> None:
    """Test mols_to_nx function with a single molecule"""
    mol = smiles_to_mols(["CCO"])[0]
    graphs = mols_to_nx([mol])
    assert len(graphs) == 1
    assert isinstance(graphs[0], nx.Graph)


def test_mols_to_nx_multiple_molecules() -> None:
    """Test mols_to_nx function with multiple molecules"""
    mol1 = smiles_to_mols(["CCO"])[0]
    mol2 = smiles_to_mols(["CCN"])[0]
    graphs = mols_to_nx([mol1, mol2])
    assert len(graphs) == 2
    assert all(isinstance(g, nx.Graph) for g in graphs)


def test_mols_to_nx_node_features() -> None:
    """Test mols_to_nx function with node features"""
    mol = smiles_to_mols(["CCO"])[0]
    graphs = mols_to_nx([mol])
    assert len(graphs[0].nodes) == mol.GetNumAtoms()
    # Check that the symbol is in the node features (label being the key)
    assert all("label" in node_data for node, node_data in graphs[0].nodes(data=True))


def test_mols_to_nx_edge_features() -> None:
    """Test mols_to_nx function with edge features"""
    mol = smiles_to_mols(["CCO"])[0]
    graphs = mols_to_nx([mol])
    assert len(graphs[0].edges) == mol.GetNumBonds()
    # Check that the bond type is in the edge features (label being the key)
    assert all(
        "label" in bond_type
        for node_start, node_end, bond_type in graphs[0].edges(data=True)
    )


def test_mols_to_nx_empty_list() -> None:
    """Test mols_to_nx function with an empty list"""
    graphs = mols_to_nx([])
    assert len(graphs) == 0

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_cc_utils.py: test functions for cc_utils.py
"""

from collections import defaultdict
from typing import Tuple, List, FrozenSet, Dict, Any

import pytest
import torch
import numpy as np
import networkx as nx
from rdkit import Chem
from easydict import EasyDict
from toponetx.classes.combinatorial_complex import CombinatorialComplex

from ccsd.src.evaluation.mmd import gaussian_emd
from ccsd.src.utils.mol_utils import mols_to_nx
from ccsd.src.utils.graph_utils import pad_adjs
from ccsd.src.utils.cc_utils import (
    get_cells,
    cc_from_incidence,
    create_incidence_1_2,
    get_rank2_dim,
    get_mol_from_x_adj,
    get_all_mol_rings,
    mols_to_cc,
    CC_to_incidence_matrices,
    ccs_to_mol,
    get_N_from_nb_edges,
    get_N_from_rank2,
    get_rank2_flags,
    mask_rank2,
    gen_noise_rank2,
    pad_rank2,
    get_global_cc_properties,
    ccs_to_tensors,
    cc_to_tensor,
    convert_CC_to_graphs,
    convert_graphs_to_CCs,
    init_flags,
    hodge_laplacian,
    default_mask,
    pow_tensor_cc,
    is_empty_cc,
    rank2_distrib_worker,
    rank2_distrib_stats,
    eval_CC_list,
    load_cc_eval_settings,
    adj_to_hodgedual,
    hodgedual_to_adj,
    get_hodge_adj_flags,
    mask_hodge_adjs,
    get_all_paths_from_single_node,
    get_all_paths_from_nodes,
    path_based_lift_CC,
)


def test_get_cells() -> None:
    """Test get_cells function."""
    N = 4
    d_min = 2
    d_max = 3

    # all_combinations, dic_set, dic_int, all_edges, dic_edge, dic_int_edge
    result = get_cells(N, d_min, d_max)
    expected_result = (
        [
            frozenset({0, 1}),
            frozenset({0, 2}),
            frozenset({0, 3}),
            frozenset({1, 2}),
            frozenset({1, 3}),
            frozenset({2, 3}),
            frozenset({0, 1, 2}),
            frozenset({0, 1, 3}),
            frozenset({0, 2, 3}),
            frozenset({1, 2, 3}),
        ],
        {
            frozenset({0, 1}): 0,
            frozenset({0, 2}): 1,
            frozenset({0, 3}): 2,
            frozenset({1, 2}): 3,
            frozenset({1, 3}): 4,
            frozenset({2, 3}): 5,
            frozenset({0, 1, 2}): 6,
            frozenset({0, 1, 3}): 7,
            frozenset({0, 2, 3}): 8,
            frozenset({1, 2, 3}): 9,
        },
        defaultdict(
            list,
            {
                0: [0, 1, 2, 6, 7, 8],
                1: [0, 3, 4, 6, 7, 9],
                2: [1, 3, 5, 6, 8, 9],
                3: [2, 4, 5, 7, 8, 9],
            },
        ),
        [
            frozenset({0, 1}),
            frozenset({0, 2}),
            frozenset({0, 3}),
            frozenset({1, 2}),
            frozenset({1, 3}),
            frozenset({2, 3}),
        ],
        {
            frozenset({0, 1}): 0,
            frozenset({0, 2}): 1,
            frozenset({0, 3}): 2,
            frozenset({1, 2}): 3,
            frozenset({1, 3}): 4,
            frozenset({2, 3}): 5,
        },
        defaultdict(list, {0: [0, 1, 2], 1: [0, 3, 4], 2: [1, 3, 5], 3: [2, 4, 5]}),
    )
    assert all(
        obj == expected_obj for obj, expected_obj in zip(result, expected_result)
    )


def test_create_incidence_1_2() -> None:
    """Test create_incidence_1_2 function."""
    d_min = 3
    d_max = 4
    X = np.array([[np.random.random() for _ in range(10)] for _ in range(5)])
    X = torch.tensor(X, dtype=torch.float32)
    N = X.shape[0]  # 5
    A = torch.zeros((N, N), dtype=torch.float32)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 0)]:
        A[i, j] = 1.0
        A[j, i] = 1.0
    two_rank_cells = {frozenset((0, 1, 2, 3)): {}, frozenset((0, 3, 4)): {}}
    F = create_incidence_1_2(N, A, d_min, d_max, two_rank_cells)
    expected_F = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert (F == expected_F).all()


@pytest.fixture
def create_incidence_1_2_test() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create incidence matrices for testing purposes.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: incidence matrices
    """
    N = 5
    nb_feat = 10
    d_min = 3
    d_max = 4

    X = np.array([[np.random.random() for _ in range(nb_feat)] for _ in range(N)])
    X = torch.tensor(X, dtype=torch.float32)
    A = torch.zeros((N, N), dtype=torch.float32)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 0)]:
        A[i, j] = 1.0
        A[j, i] = 1.0
    two_rank_cells = {frozenset((0, 1, 2, 3)): {}, frozenset((0, 3, 4)): {}}
    F = create_incidence_1_2(N, A, d_min, d_max, two_rank_cells)
    F = torch.tensor(F, dtype=torch.float32)
    return (X, A, F)


@pytest.fixture
def create_incidence_1_2_advanced_test() -> (
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Create incidence matrices for testing purposes.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: incidence matrices
    """
    N = 5
    nb_feat = 10
    nb_adj_feat = 3
    nb_rank2_feat = 4
    d_min = 3
    d_max = 4

    X = np.array([[np.random.random() for _ in range(nb_feat)] for _ in range(N)])
    X = torch.tensor(X, dtype=torch.float32)
    A = torch.zeros((N, N, nb_adj_feat), dtype=torch.float32)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 0)]:
        A[i, j] = torch.rand(nb_adj_feat)
        A[j, i] = A[i, j]
    random_val1, random_val2 = torch.rand(nb_rank2_feat), torch.rand(nb_rank2_feat)
    two_rank_cells = {
        frozenset((0, 1, 2, 3)): {
            f"label_{k}": random_val1[k].item() for k in range(nb_rank2_feat)
        },
        frozenset((0, 3, 4)): {
            f"label_{k}": random_val2[k].item() for k in range(nb_rank2_feat)
        },
    }
    F = create_incidence_1_2(N, A, d_min, d_max, two_rank_cells)
    F = torch.tensor(F, dtype=torch.float32)
    return (X, A, F)


@pytest.fixture
def create_incidence_1_2_test_tiny() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create incidence matrices for testing purposes.
    Tiny version with smaller rank-2 incidence matrix.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: incidence matrices
    """
    N = 4
    nb_feat = 3
    d_min = 2
    d_max = 3

    X = np.array([[np.random.random() for _ in range(nb_feat)] for _ in range(N)])
    X = torch.tensor(X, dtype=torch.float32)
    A = torch.zeros((N, N), dtype=torch.float32)
    for i, j in [(0, 1), (1, 2), (2, 3), (0, 3)]:
        A[i, j] = 1.0
        A[j, i] = 1.0
    two_rank_cells = {frozenset((0, 1, 2)): {}, frozenset((2, 3)): {}}
    F = create_incidence_1_2(N, A, d_min, d_max, two_rank_cells)
    F = torch.tensor(F, dtype=torch.float32)
    return (X, A, F)


def test_cc_from_incidence(
    create_incidence_1_2_test: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> None:
    """Test cc_from_incidence function.

    Args:
        create_incidence_1_2_test (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): incidence matrices
    """
    d_min = 3
    d_max = 4
    N = 5  # nb nodes
    nb_feat = 10

    # Check that the function returns a CombinatorialComplex object
    CC = cc_from_incidence(None, d_min, d_max, is_molecule=False)
    assert isinstance(CC, CombinatorialComplex)
    assert not (len(CC.nodes))  # check that the CC is empty

    X, A, F = create_incidence_1_2_test

    # Check the result for rank-0 CC
    incidence_matrices = [X, None, None]
    CC = cc_from_incidence(incidence_matrices, d_min, d_max)
    assert len(CC.cells.hyperedge_dict) == 1
    assert 0 in CC.cells.hyperedge_dict
    assert len(CC.cells.hyperedge_dict[0][frozenset({0})]) == (
        nb_feat + 1
    )  # with default weight
    assert len(CC.cells.hyperedge_dict[0]) == N

    # Check the result for rank-1 CC
    incidence_matrices = [X, A, None]
    CC = cc_from_incidence(incidence_matrices, d_min, d_max)
    assert len(CC.cells.hyperedge_dict) == 2
    assert 1 in CC.cells.hyperedge_dict
    assert (
        len(CC.cells.hyperedge_dict[1][frozenset({0, 1})]) == 2
    )  # with default weight
    assert len(CC.cells.hyperedge_dict[1]) == 6  # 6 edges

    # Check the result for rank-2 CC
    incidence_matrices = [X, A, F]
    CC = cc_from_incidence(incidence_matrices, d_min, d_max)
    assert len(CC.cells.hyperedge_dict) == 3
    assert 2 in CC.cells.hyperedge_dict
    assert (
        len(CC.cells.hyperedge_dict[2][frozenset({0, 3, 4})]) == 2
    )  # with default weight and "label" created from the conversion (constant to 1.0 here)
    assert len(CC.cells.hyperedge_dict[2]) == 2  # 2 higher-order cells


def test_cc_incidence_advanced_test(
    create_incidence_1_2_advanced_test: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> None:
    """Advanced test to check the correspondence between the incidence matrices and the CC
    when doing conversions.

    Args:
        create_incidence_1_2_advanced_test (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): incidence matrices
    """
    X, A, F = create_incidence_1_2_advanced_test
    d_min = 3
    d_max = 4
    # Convert to CC
    cc = cc_from_incidence([X, A, F], 3, 4, is_molecule=False)
    # Convert back to incidence matrices
    X2, A2, F2 = CC_to_incidence_matrices(cc, d_min, d_max)
    # Convert to tensor
    X2 = torch.tensor(X2, dtype=torch.float32)
    A2 = torch.tensor(A2, dtype=torch.float32)
    F2 = torch.tensor(F2, dtype=torch.float32)
    assert (X == X2).all()
    assert (A == A2).all()
    assert (F == F2).all()


def test_get_rank2_dim(
    create_incidence_1_2_test: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> None:
    """Test get_rank2_dim function.

    Args:
        create_incidence_1_2_test (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): incidence matrices
    """
    N = 5  # nb nodes
    d_min = 3
    d_max = 4

    _, _, F = create_incidence_1_2_test

    # Check the result
    assert get_rank2_dim(N, d_min, d_max) == F.shape


def test_get_mol_from_x_adj() -> None:
    """Test get_mol_from_x_adj function."""
    x = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    adj = torch.tensor([[0, 1], [1, 0]])

    mol = get_mol_from_x_adj(x, adj)

    assert isinstance(mol, Chem.Mol)
    assert Chem.MolToSmiles(mol) == "NO"


def test_get_all_mol_rings() -> None:
    """Test get_all_mol_rings function."""
    mol = Chem.MolFromSmiles("Cc1ccccc1")  # Create a molecule from a SMILES string

    rings = get_all_mol_rings(mol)
    assert rings == [frozenset({1, 2, 3, 4, 5, 6})]


def test_mols_to_cc() -> None:
    """Test mols_to_cc function."""
    mol1 = Chem.MolFromSmiles("Cc1ccccc1")
    mol2 = Chem.MolFromSmiles("c1cccc2c1CCCC2")
    mols = [mol1, mol2]

    ccs = mols_to_cc(mols)

    # Check if the output is a list of CombinatorialComplex objects
    assert isinstance(ccs, list)
    assert len(ccs) == 2
    assert all(isinstance(cc, CombinatorialComplex) for cc in ccs)
    # Check if the combinatorial complexes are the good ones
    assert ccs[0].cells.hyperedge_dict == {
        0: {
            frozenset({0}): {"symbol": 6, "weight": 1},
            frozenset({1}): {"symbol": 6, "weight": 1},
            frozenset({2}): {"symbol": 6, "weight": 1},
            frozenset({3}): {"symbol": 6, "weight": 1},
            frozenset({4}): {"symbol": 6, "weight": 1},
            frozenset({5}): {"symbol": 6, "weight": 1},
            frozenset({6}): {"symbol": 6, "weight": 1},
        },
        1: {
            frozenset({0, 1}): {"bond_type": 1.0, "weight": 1},
            frozenset({1, 2}): {"bond_type": 1.5, "weight": 1},
            frozenset({2, 3}): {"bond_type": 1.5, "weight": 1},
            frozenset({3, 4}): {"bond_type": 1.5, "weight": 1},
            frozenset({4, 5}): {"bond_type": 1.5, "weight": 1},
            frozenset({5, 6}): {"bond_type": 1.5, "weight": 1},
            frozenset({1, 6}): {"bond_type": 1.5, "weight": 1},
        },
        2: {frozenset({1, 2, 3, 4, 5, 6}): {"weight": 1}},
    }
    assert ccs[1].cells.hyperedge_dict == {
        0: {
            frozenset({0}): {"symbol": 6, "weight": 1},
            frozenset({1}): {"symbol": 6, "weight": 1},
            frozenset({2}): {"symbol": 6, "weight": 1},
            frozenset({3}): {"symbol": 6, "weight": 1},
            frozenset({4}): {"symbol": 6, "weight": 1},
            frozenset({5}): {"symbol": 6, "weight": 1},
            frozenset({6}): {"symbol": 6, "weight": 1},
            frozenset({7}): {"symbol": 6, "weight": 1},
            frozenset({8}): {"symbol": 6, "weight": 1},
            frozenset({9}): {"symbol": 6, "weight": 1},
        },
        1: {
            frozenset({0, 1}): {"bond_type": 1.5, "weight": 1},
            frozenset({1, 2}): {"bond_type": 1.5, "weight": 1},
            frozenset({2, 3}): {"bond_type": 1.5, "weight": 1},
            frozenset({3, 4}): {"bond_type": 1.5, "weight": 1},
            frozenset({4, 5}): {"bond_type": 1.5, "weight": 1},
            frozenset({5, 6}): {"bond_type": 1.0, "weight": 1},
            frozenset({6, 7}): {"bond_type": 1.0, "weight": 1},
            frozenset({8, 7}): {"bond_type": 1.0, "weight": 1},
            frozenset({8, 9}): {"bond_type": 1.0, "weight": 1},
            frozenset({0, 5}): {"bond_type": 1.5, "weight": 1},
            frozenset({9, 4}): {"bond_type": 1.0, "weight": 1},
        },
        2: {
            frozenset({0, 1, 2, 3, 4, 5}): {"weight": 1},
            frozenset({4, 5, 6, 7, 8, 9}): {"weight": 1},
        },
    }


def test_CC_to_incidence_matrices(
    create_incidence_1_2_test: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> None:
    """Test CC_to_incidence_matrices function.

    Args:
        create_incidence_1_2_test (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): incidence matrices
    """
    d_min = 3
    d_max = 4

    X, A, F = create_incidence_1_2_test

    # Convert tensors to arrays to compare with the result of CC_to_incidence_matrices
    X, A, F = X.numpy(), A.numpy(), F.numpy()

    # Check for empty CC
    incidence_matrices = [None, None, None]
    cc = cc_from_incidence(incidence_matrices, d_min, d_max)
    expected_incidence_matrices = CC_to_incidence_matrices(cc, d_min, d_max)
    assert (
        (incidence_matrices[0] == expected_incidence_matrices[0]).all()
        if X is not None
        else (
            (expected_incidence_matrices[0] is None)
            or not (expected_incidence_matrices[0].size)
        )
    )
    assert (
        (incidence_matrices[1] == expected_incidence_matrices[1]).all()
        if A is not None
        else (
            (expected_incidence_matrices[1] is None)
            or not (expected_incidence_matrices[1].size)
        )
    )
    assert (
        (incidence_matrices[2] == expected_incidence_matrices[2]).all()
        if F is not None
        else (
            (expected_incidence_matrices[2] is None)
            or not (expected_incidence_matrices[2].size)
        )
    )

    # Check for rank-0 CC
    incidence_matrices = [X, None, None]
    cc = cc_from_incidence(incidence_matrices, d_min, d_max)
    expected_incidence_matrices = CC_to_incidence_matrices(cc, d_min, d_max)
    assert (
        (incidence_matrices[0] == expected_incidence_matrices[0]).all()
        if X is not None
        else (
            (expected_incidence_matrices[0] is None)
            or not (expected_incidence_matrices[0].size)
        )
    )
    assert (
        (incidence_matrices[1] == expected_incidence_matrices[1]).all()
        if A is not None
        else (
            (expected_incidence_matrices[1] is None)
            or not (expected_incidence_matrices[1].size)
        )
    )
    assert (
        (incidence_matrices[2] == expected_incidence_matrices[2]).all()
        if F is not None
        else (
            (expected_incidence_matrices[2] is None)
            or not (expected_incidence_matrices[2].size)
        )
    )

    # Check for rank-1 CC
    incidence_matrices = [X, A, None]
    cc = cc_from_incidence(incidence_matrices, d_min, d_max)
    expected_incidence_matrices = CC_to_incidence_matrices(cc, d_min, d_max)
    assert (
        (incidence_matrices[0] == expected_incidence_matrices[0]).all()
        if X is not None
        else (
            (expected_incidence_matrices[0] is None)
            or not (expected_incidence_matrices[0].size)
        )
    )
    assert (
        (incidence_matrices[1] == expected_incidence_matrices[1]).all()
        if A is not None
        else (
            (expected_incidence_matrices[1] is None)
            or not (expected_incidence_matrices[1].size)
        )
    )
    assert (
        (incidence_matrices[2] == expected_incidence_matrices[2]).all()
        if F is not None
        else (
            (expected_incidence_matrices[2] is None)
            or not (expected_incidence_matrices[2].size)
        )
    )

    # Check for rank-2 CC
    incidence_matrices = [X, A, F]
    cc = cc_from_incidence(incidence_matrices, d_min, d_max)
    expected_incidence_matrices = CC_to_incidence_matrices(cc, d_min, d_max)
    assert (
        (incidence_matrices[0] == expected_incidence_matrices[0]).all()
        if X is not None
        else (
            (expected_incidence_matrices[0] is None)
            or not (expected_incidence_matrices[0].size)
        )
    )
    assert (
        (incidence_matrices[1] == expected_incidence_matrices[1]).all()
        if A is not None
        else (
            (expected_incidence_matrices[1] is None)
            or not (expected_incidence_matrices[1].size)
        )
    )
    assert (
        (incidence_matrices[2] == expected_incidence_matrices[2]).all()
        if F is not None
        else (
            (expected_incidence_matrices[2] is None)
            or not (expected_incidence_matrices[2].size)
        )
    )


def test_ccs_to_mol() -> None:
    """Test ccs_to_mol function."""
    mol1 = Chem.MolFromSmiles("Cc1ccccc1")
    mol2 = Chem.MolFromSmiles("c1cccc2c1CCCC2")
    mols = [mol1, mol2]
    ccs = mols_to_cc(mols)
    new_mols = ccs_to_mol(ccs)
    assert isinstance(new_mols, list)
    assert len(new_mols) == 2
    assert all(isinstance(mol, Chem.Mol) for mol in new_mols)
    assert Chem.MolToSmiles(new_mols[0]) == "Cc1ccccc1"
    assert Chem.MolToSmiles(new_mols[1]) in ("c1cccc2c1CCCC2", "c1ccc2c(c1)CCCC2")


def test_get_N_from_nb_edges() -> None:
    """Test get_N_from_nb_edges function."""
    assert get_N_from_nb_edges(1) == 2
    assert get_N_from_nb_edges(6) == 4
    assert get_N_from_nb_edges(300) == 25


def test_get_N_from_rank2(
    create_incidence_1_2_test: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> None:
    """Test get_N_from_rank2 function.

    Args:
        create_incidence_1_2_test (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): incidence matrices
    """
    X, _, F = create_incidence_1_2_test
    N = X.shape[0]
    assert get_N_from_rank2(F) == N


def test_get_rank2_flags() -> None:
    """Test get_rank2_flags function."""
    N = 4
    d_min = 3
    d_max = 3
    rank2 = torch.tensor([[[1, 1, 1, 1] for _ in range(6)]])
    flags = torch.tensor([[1, 1, 0, 1]])  # remove node 2
    flags_left, flags_right = get_rank2_flags(rank2, N, d_min, d_max, flags)
    assert flags_left.shape == (1, 6)
    assert flags_right.shape == (1, 4)
    assert torch.allclose(flags_left, torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]))
    assert torch.allclose(flags_right, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))


def test_mask_rank2() -> None:
    """Test mask_rank2 function."""
    N = 4
    d_min = 3
    d_max = 3
    nb_edges = 6  # number of different possible edges with N=4
    rank2 = torch.tensor([[[1, 1, 1, 1] for _ in range(nb_edges)]], dtype=torch.float32)
    flags = torch.tensor([[1, 1, 0, 1]], dtype=torch.float32)  # remove node 2
    masked = mask_rank2(rank2, N, d_min, d_max, flags)
    assert masked.shape == (1, nb_edges, N)
    assert torch.allclose(
        masked,
        torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ]
        ),
    )

    # Test with channels
    nb_channels = 3
    c_rank2 = torch.cat([rank2 for _ in range(nb_channels)], dim=0).unsqueeze(0)
    c_masked = mask_rank2(c_rank2, N, d_min, d_max, flags)
    expected_masked = torch.cat([masked for _ in range(nb_channels)], dim=0).unsqueeze(
        0
    )
    assert c_masked.shape == (1, nb_channels, nb_edges, N)
    assert torch.allclose(expected_masked, c_masked)


def test_gen_noise_rank2() -> None:
    """Test gen_noise_rank2 function."""
    N = 4
    d_min = 3
    d_max = 3
    x = torch.rand((1, 6, 4))
    noise = gen_noise_rank2(x, N, d_min, d_max)
    assert noise.shape == x.shape


def test_pad_rank2() -> None:
    """Test pad_rank2 function."""
    N = 4
    d_min = 3
    d_max = 3
    rank2 = np.array([[1, 1, 1, 1] for _ in range(6)], dtype=np.float32)

    assert (pad_rank2(rank2, N, d_min=d_min, d_max=d_max) == rank2).all()
    assert (
        pad_rank2(rank2, N + 1, d_min=d_min, d_max=d_max)
        == np.array(
            [
                [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()
    with pytest.raises(ValueError):
        pad_rank2(rank2, N - 1, d_min=d_min, d_max=d_min)


def test_get_global_cc_properties() -> None:
    """Test get_global_cc_properties function."""
    mol1 = Chem.MolFromSmiles("Cc1ccccc1")
    mol2 = Chem.MolFromSmiles("c1cccc2c1CCCC2")
    mol3 = Chem.MolFromSmiles("C1CC1")
    mols = [mol1, mol2, mol3]
    ccs = mols_to_cc(mols)
    max_node_num, d_min, d_max = get_global_cc_properties(ccs)
    assert max_node_num == 10
    assert d_min == 3
    assert d_max == 6


def test_ccs_to_tensors() -> None:
    """Test ccs_to_tensors function."""
    mol1 = Chem.MolFromSmiles("Cc1ccccc1")
    mol2 = Chem.MolFromSmiles("c1cccc2c1CCCC2")
    mol3 = Chem.MolFromSmiles("C1CC1")
    mols = [mol1, mol2, mol3]
    ccs = mols_to_cc(mols)
    # Get the global properties
    max_node_num, d_min, d_max = get_global_cc_properties(ccs)
    # Get the incidence matrices
    _, A1, F1 = CC_to_incidence_matrices(ccs[0], d_min=d_min, d_max=d_max)
    _, A2, F2 = CC_to_incidence_matrices(ccs[1], d_min=d_min, d_max=d_max)
    _, A3, F3 = CC_to_incidence_matrices(ccs[2], d_min=d_min, d_max=d_max)
    # Pad the tensors
    A1 = pad_adjs(A1, max_node_num)
    F1 = pad_rank2(F1, max_node_num, d_min=d_min, d_max=d_max)
    A2 = pad_adjs(A2, max_node_num)
    F2 = pad_rank2(F2, max_node_num, d_min=d_min, d_max=d_max)
    A3 = pad_adjs(A3, max_node_num)
    F3 = pad_rank2(F3, max_node_num, d_min=d_min, d_max=d_max)
    # Stack the arrays and transform into tensors
    A = torch.tensor(np.stack([A1, A2, A3]), dtype=torch.float32)
    F = torch.tensor(np.stack([F1, F2, F3]), dtype=torch.float32)
    # Compare to the result of ccs_to_tensors
    As, Fs = ccs_to_tensors(ccs, max_node_num, d_min, d_max)
    assert As.shape == A.shape
    assert As.shape == (3, 10, 10)
    assert torch.allclose(As, A)
    assert Fs.shape == F.shape
    assert Fs.shape == (3, 45, 792)
    assert torch.allclose(Fs, F)


def test_cc_to_tensor() -> None:
    """Test cc_to_tensor function."""
    mol = Chem.MolFromSmiles("C1CC1")
    cc = mols_to_cc([mol])[0]
    As, Fs = cc_to_tensor(cc, max_node_num=4, d_min=3, d_max=4)
    assert As.shape == (4, 4)
    assert Fs.shape == (6, 5)
    assert torch.allclose(
        As,
        torch.tensor(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    assert torch.allclose(
        Fs,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_convert_CC_to_graphs() -> None:
    """Test convert_CC_to_graphs function."""
    mol1 = Chem.MolFromSmiles("Cc1ccccc1")
    mol2 = Chem.MolFromSmiles("C1CC1")
    mols = [mol1, mol2]
    ccs = mols_to_cc(mols)
    graphs = convert_CC_to_graphs(ccs)

    assert isinstance(graphs, list)
    assert len(graphs) == 2
    assert all(isinstance(g, nx.Graph) for g in graphs)
    assert graphs[0].number_of_nodes() == 7
    assert graphs[0].number_of_edges() == 7
    assert graphs[1].number_of_nodes() == 3
    assert graphs[1].number_of_edges() == 3
    assert graphs[0]._node == {
        0: {"symbol": 6, "weight": 1},
        1: {"symbol": 6, "weight": 1},
        2: {"symbol": 6, "weight": 1},
        3: {"symbol": 6, "weight": 1},
        4: {"symbol": 6, "weight": 1},
        5: {"symbol": 6, "weight": 1},
        6: {"symbol": 6, "weight": 1},
    }
    assert graphs[1]._node == {
        0: {"symbol": 6, "weight": 1},
        1: {"symbol": 6, "weight": 1},
        2: {"symbol": 6, "weight": 1},
    }
    assert graphs[0]._adj == {
        0: {1: {"bond_type": 1.0, "weight": 1}},
        1: {
            0: {"bond_type": 1.0, "weight": 1},
            2: {"bond_type": 1.5, "weight": 1},
            6: {"bond_type": 1.5, "weight": 1},
        },
        2: {1: {"bond_type": 1.5, "weight": 1}, 3: {"bond_type": 1.5, "weight": 1}},
        3: {2: {"bond_type": 1.5, "weight": 1}, 4: {"bond_type": 1.5, "weight": 1}},
        4: {3: {"bond_type": 1.5, "weight": 1}, 5: {"bond_type": 1.5, "weight": 1}},
        5: {4: {"bond_type": 1.5, "weight": 1}, 6: {"bond_type": 1.5, "weight": 1}},
        6: {5: {"bond_type": 1.5, "weight": 1}, 1: {"bond_type": 1.5, "weight": 1}},
    }
    assert graphs[1]._adj == {
        0: {1: {"bond_type": 1.0, "weight": 1}, 2: {"bond_type": 1.0, "weight": 1}},
        1: {0: {"bond_type": 1.0, "weight": 1}, 2: {"bond_type": 1.0, "weight": 1}},
        2: {1: {"bond_type": 1.0, "weight": 1}, 0: {"bond_type": 1.0, "weight": 1}},
    }


def test_convert_graphs_to_CCs() -> None:
    """Test convert_graphs_to_CCs function."""
    mol1 = Chem.MolFromSmiles("Cc1ccccc1")
    mol2 = Chem.MolFromSmiles("c1cccc2c1CCCC2")
    mols = [mol1, mol2]
    graphs = mols_to_nx(mols)

    # Convert to CCS
    ccs = convert_graphs_to_CCs(graphs)
    assert isinstance(ccs, list)
    assert len(ccs) == 2
    assert all(isinstance(cc, CombinatorialComplex) for cc in ccs)
    assert ccs[0].cells.hyperedge_dict[0][frozenset({0})] == {"label": "C", "weight": 1}
    assert ccs[0].cells.hyperedge_dict[1][frozenset({0, 1})] == {
        "label": 1,
        "weight": 1,
    }

    # Convert to CCS but for molecules
    ccs = convert_graphs_to_CCs(graphs, is_molecule=True)
    assert ccs[0].cells.hyperedge_dict[0][frozenset({0})] == {"symbol": 6, "weight": 1}
    assert ccs[0].cells.hyperedge_dict[1][frozenset({0, 1})] == {
        "bond_type": 1.0,
        "weight": 1,
    }


def create_sample_graphs(
    num_graphs: int, num_nodes: int, num_edges: int
) -> List[nx.Graph]:
    """Create a list of sample graphs.

    Args:
        num_graphs (int): number of graphs to create
        num_nodes (int): number of nodes in each graph
        num_edges (int): number of edges in each graph

    Returns:
        List[nx.Graph]: a list of sample graphs
    """
    graph_list = []
    for _ in range(num_graphs):
        graph = nx.gnm_random_graph(num_nodes, num_edges)
        graph_list.append(graph)
    return graph_list


def create_sample_ccs(
    num_ccs: int, num_nodes: int, num_edges: int
) -> List[CombinatorialComplex]:
    """Create a list of sample combinatorial complexes.

    Args:
        num_ccs (int): number of combinatorial complexes to create
        num_nodes (int): number of nodes in each combinatorial complex
        num_edges (int): number of edges in each combinatorial complex

    Returns:
        List[CombinatorialComplex]: a list of sample combinatorial complexes
    """
    return convert_graphs_to_CCs(create_sample_graphs(num_ccs, num_nodes, num_edges))


def test_init_flags() -> None:
    """Test the init_flags function (return node flags for each graphs/ccs of the batch)."""

    # Test for graphs
    num_graphs = 10
    num_nodes = 8
    num_edges = 5
    batch_size = 3
    config = EasyDict(
        {
            "data": {
                "batch_size": batch_size,
                "max_node_num": num_nodes,
                "d_min": 3,
                "d_max": 4,
            }
        }
    )
    graph_list = create_sample_graphs(num_graphs, num_nodes, num_edges)
    flags = init_flags(graph_list, config, batch_size=batch_size)
    assert flags.shape == (batch_size, num_nodes)
    assert torch.all((flags == 0) | (flags == 1)).item()

    # Test for combinatorial complexes
    num_ccs = 10
    num_nodes = 8
    num_edges = 5
    batch_size = 3
    config = EasyDict(
        {
            "data": {
                "batch_size": batch_size,
                "max_node_num": num_nodes,
                "d_min": 3,
                "d_max": 4,
            }
        }
    )
    cc_list = create_sample_ccs(num_ccs, num_nodes, num_edges)
    flags = init_flags(cc_list, config, batch_size=batch_size, is_cc=True)
    assert flags.shape == (batch_size, num_nodes)
    assert torch.all((flags == 0) | (flags == 1)).item()


def test_hodge_laplacian(
    create_incidence_1_2_test: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> None:
    """Test the hodge_laplacian function.

    Args:
        create_incidence_1_2_test (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): incidence matrices
    """
    _, _, F = create_incidence_1_2_test
    F = F.unsqueeze(0)  # add batch dimension
    H = hodge_laplacian(F)
    expected_H = torch.tensor(
        [
            [
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    assert H.shape == (1, F.shape[1], F.shape[1])
    assert torch.allclose(H, expected_H)


def test_default_mask() -> None:
    """Test the default_mask function."""
    mask = default_mask(3)
    expected_mask = torch.tensor(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(mask, expected_mask)


def test_pow_tensor_cc(
    create_incidence_1_2_test_tiny: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> None:
    """Test the pow_tensor_cc function.

    Args:
        create_incidence_1_2_test_tiny (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): incidence matrices
    """
    _, _, F = create_incidence_1_2_test_tiny
    F = F.unsqueeze(0)  # add batch dimension
    c = 2
    # Test without hodge mask
    res = pow_tensor_cc(F, c, hodge_mask=None)
    expected_res = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ]
        ],
        dtype=torch.float32,
    )
    assert res.shape == (1, c, F.shape[1], F.shape[2])
    assert torch.allclose(res, expected_res)

    # Test with hodge mask
    hodge_mask = default_mask(F.shape[1])
    res = pow_tensor_cc(F, c, hodge_mask=hodge_mask)
    expected_res = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ]
        ],
        dtype=torch.float32,
    )
    assert res.shape == (1, c, F.shape[1], F.shape[2])
    assert torch.allclose(res, expected_res)


def test_is_empty_cc_empty_complex() -> None:
    """Test is_empty_cc on an empty complex."""
    cc = CombinatorialComplex()  # empty combinatorial complex
    assert is_empty_cc(cc) is True


def test_is_empty_cc_non_empty_complex() -> None:
    """Test is_empty_cc on a non-empty complex."""
    cc = CombinatorialComplex()
    cc.add_cell((0,), rank=0)  # make it non-empty
    assert is_empty_cc(cc) is False


def test_rank2_distrib_worker(
    create_incidence_1_2_test: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> None:
    """Test rank2_distrib_worker on a CombinatorialComplex with rank-2 cells.

    Args:
        create_incidence_1_2_test: A tuple of torch.Tensors representing the incidence matrices of a CombinatorialComplex.
    """
    X, A, F = create_incidence_1_2_test
    d_min = 3
    d_max = 4
    cc = cc_from_incidence([X, A, F], d_min=d_min, d_max=d_max)

    result = rank2_distrib_worker(cc, d_min, d_max)
    assert isinstance(result, np.ndarray)
    assert result.shape == (d_max - d_min + 1,)
    assert np.all(result >= 0)
    assert np.sum(result) == len(cc.cells.hyperedge_dict.get(2, {}))
    assert (result == np.array([1.0, 1.0])).all()


def test_rank2_distrib_stats(
    create_incidence_1_2_test: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> None:
    """Test rank2_distrib_stats on a CombinatorialComplex with rank-2 cells.

    Args:
        create_incidence_1_2_test (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple of torch.Tensors
    """
    # Create some test CombinatorialComplex instances
    X, A, F = create_incidence_1_2_test
    d_min = 3
    d_max = 4
    # CCs with different rank2_distrib:
    cc1 = cc_from_incidence([X, A, F], d_min=d_min, d_max=d_max)  # [1., 1.]
    cc2 = cc_from_incidence([X, A, F], d_min=d_min, d_max=d_max)  # [2., 1.]
    cc3 = cc_from_incidence([X, A, F], d_min=d_min, d_max=d_max)  # [3., 1.]
    cc4 = cc_from_incidence([X, A, F], d_min=d_min, d_max=d_max)  # [3., 2.]

    # Add some cells to the CombinatorialComplex instances
    cc2.add_cell(frozenset((0, 1, 2)), rank=2)
    cc3.add_cell(frozenset((0, 1, 2)), rank=2)
    cc4.add_cell(frozenset((0, 1, 2)), rank=2)
    cc3.add_cell(frozenset((0, 2, 3)), rank=2)
    cc4.add_cell(frozenset((0, 2, 3)), rank=2)
    cc4.add_cell(frozenset((1, 2, 3, 4)), rank=2)

    # Create the lists of CombinatorialComplex instances
    cc_ref_list = [cc1, cc2]
    cc_pred_list = [cc3, cc4]

    # Compute the statistics
    result = rank2_distrib_stats(cc_ref_list, cc_pred_list, d_min, d_max)
    assert isinstance(result, float)
    assert result == 0.008230171666159913


def test_eval_CC_list(
    create_incidence_1_2_test: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> None:
    """Test eval_CC_list with default methods and kernels.

    Args:
        create_incidence_1_2_test: A tuple of torch.Tensors representing the incidence matrices of a CombinatorialComplex.
    """
    # Create some test CombinatorialComplex instances
    X, A, F = create_incidence_1_2_test
    d_min = 3
    d_max = 4
    # CCs with different rank2_distrib:
    cc1 = cc_from_incidence([X, A, F], d_min=d_min, d_max=d_max)  # [1., 1.]
    cc2 = cc_from_incidence([X, A, F], d_min=d_min, d_max=d_max)  # [2., 1.]
    cc3 = cc_from_incidence([X, A, F], d_min=d_min, d_max=d_max)  # [3., 1.]
    cc4 = cc_from_incidence([X, A, F], d_min=d_min, d_max=d_max)  # [3., 2.]

    # Add some cells to the CombinatorialComplex instances
    cc2.add_cell(frozenset((0, 1, 2)), rank=2)
    cc3.add_cell(frozenset((0, 1, 2)), rank=2)
    cc4.add_cell(frozenset((0, 1, 2)), rank=2)
    cc3.add_cell(frozenset((0, 2, 3)), rank=2)
    cc4.add_cell(frozenset((0, 2, 3)), rank=2)
    cc4.add_cell(frozenset((1, 2, 3, 4)), rank=2)

    # Create the lists of CombinatorialComplex instances
    cc_ref_list = [cc1, cc2]
    cc_pred_list = [cc3, cc4]

    # Call the function with default methods and kernels
    result = eval_CC_list(
        cc_ref_list,
        cc_pred_list,
        d_min,
        d_max,
        methods=["rank2_distrib"],
        kernels={
            "rank2_distrib": gaussian_emd,
        },
    )
    assert isinstance(result, dict)
    assert "rank2_distrib" in result
    assert isinstance(result["rank2_distrib"], float)
    assert result["rank2_distrib"] == 0.00823  # statistics round to 6 digits

    # Try calling the function with an invalid method
    with pytest.raises(KeyError):
        eval_CC_list(
            cc_ref_list,
            cc_pred_list,
            d_min,
            d_max,
            methods=["invalid_method"],
            kernels={},
        )


def test_load_cc_eval_settings() -> None:
    """Test load_cc_eval_settings."""
    # Call the function to get the output
    methods, kernels = load_cc_eval_settings()

    # Define the expected values for methods and kernels
    expected_methods = ["rank2_distrib"]
    expected_kernels = {
        "rank2_distrib": gaussian_emd,
    }

    # Check if the returned values match the expected values
    assert (
        methods == expected_methods
    ), f"Expected methods: {expected_methods}, but got {methods}"
    assert (
        kernels == expected_kernels
    ), f"Expected kernels: {expected_kernels}, but got {kernels}"


@pytest.fixture
def create_batch_channel_adj_tensor() -> torch.Tensor:
    """Create a batch of channel adjacency tensors for testing purposes.
    The adjacency matrices are 4x4 matrices with 1 channel and 1 batch dimension and are symmetric.

    Returns:
        torch.Tensor: A batch of channel adjacency tensors.
    """
    batch_size = 1
    channels = 1
    # Starting from a 4x4 adjacency matrix
    adj = torch.tensor(
        [[0, 2, 3, 4], [0, 0, 7, 8], [0, 0, 0, 12], [0, 0, 0, 0]],
        dtype=torch.float32,
    )
    adj = adj + adj.transpose(-1, -2)
    # Add channels
    adj = adj.unsqueeze(0)
    adj = torch.cat([adj for _ in range(channels)], dim=0)
    # Add batch dimension
    adj = adj.unsqueeze(0)
    adj = torch.cat([adj for _ in range(batch_size)], dim=0)

    return adj


def test_adj_to_hodgedual(create_batch_channel_adj_tensor: torch.Tensor) -> None:
    """Test adj_to_hodgedual.

    Args:
        create_batch_channel_adj_tensor: A batch of channel adjacency tensors.
    """
    hodgedual = adj_to_hodgedual(create_batch_channel_adj_tensor)
    expected_hodgedual = torch.tensor(
        [
            [
                [
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 7.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 8.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 12.0],
                ]
            ]
        ]
    )
    assert torch.allclose(hodgedual, expected_hodgedual)


def test_hodgedual_to_adj(create_batch_channel_adj_tensor: torch.Tensor) -> None:
    """Test hodgedual_to_adj.

    Args:
        create_batch_channel_adj_tensor (torch.Tensor): A batch of channel adjacency tensors.
    """
    original_adj = create_batch_channel_adj_tensor
    hodgedual = adj_to_hodgedual(original_adj)
    adj = hodgedual_to_adj(hodgedual)
    assert torch.allclose(original_adj, adj)


def test_get_hodge_adj_flags() -> None:
    """Test get_hodge_adj_flags function."""
    N = 4
    nb_edges = 6  # number of different possible edges with N=4
    hodge_adj = torch.ones((nb_edges, nb_edges), dtype=torch.float32).unsqueeze(0)
    flags = torch.tensor([[1, 1, 0, 1]])  # remove node 2
    flags_hodge = get_hodge_adj_flags(hodge_adj, flags)
    assert flags_hodge.shape == (1, nb_edges)
    assert torch.allclose(flags_hodge, torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]))


def test_mask_hodge_adjs() -> None:
    """Test mask_hodge_adjs function."""
    N = 4
    nb_edges = 6  # number of different possible edges with N=4
    hodge_adj = torch.ones((nb_edges, nb_edges), dtype=torch.float32).unsqueeze(0)
    flags = torch.tensor([[1, 1, 0, 1]], dtype=torch.float32)  # remove node 2
    masked = mask_hodge_adjs(hodge_adj, flags)
    assert masked.shape == (1, nb_edges, nb_edges)
    assert torch.allclose(
        masked,
        torch.tensor(
            [
                [
                    [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        ),
    )

    # Test with channels
    nb_channels = 3
    c_hodge_adj = torch.cat([hodge_adj for _ in range(nb_channels)], dim=0).unsqueeze(0)
    c_masked = mask_hodge_adjs(c_hodge_adj, flags)
    expected_masked = torch.cat([masked for _ in range(nb_channels)], dim=0).unsqueeze(
        0
    )
    assert c_masked.shape == (1, nb_channels, nb_edges, nb_edges)
    assert torch.allclose(expected_masked, c_masked)


@pytest.fixture
def create_edge_dict_graph() -> Dict[FrozenSet[int], Dict[str, Any]]:
    """Create a graph with 4 nodes and 6 edges.

    Returns:
        Dict[FrozenSet[int], Dict[str, Any]]: A graph with 4 nodes and 6 edges.
    """
    return {
        frozenset({0, 1}): {"weight": 1.0},
        frozenset({0, 2}): {"weight": 2.0},
        frozenset({0, 3}): {"weight": 3.0},
        frozenset({1, 2}): {"weight": 4.0},
        frozenset({1, 3}): {"weight": 5.0},
        frozenset({2, 3}): {"weight": 6.0},
    }


def test_get_all_paths_from_single_node(
    create_edge_dict_graph: Dict[FrozenSet[int], Dict[str, Any]]
) -> None:
    """Test get_all_paths_from_single_node function.

    Args:
        create_edge_dict_graph: A fixture graph with 4 nodes and 6 edges.
    """
    g = defaultdict(list)
    for u, v in create_edge_dict_graph.keys():
        g[u].append(v)
        g[v].append(u)
    paths = get_all_paths_from_single_node(0, g, 2)
    assert paths == set([frozenset([0, 1]), frozenset([0, 2]), frozenset([0, 3])])
    paths = get_all_paths_from_single_node(0, g, 3)
    assert paths == set(
        [
            frozenset([0, 1, 2]),
            frozenset([0, 1, 3]),
            frozenset([0, 2, 1]),
            frozenset([0, 2, 3]),
        ]
    )


def test_get_all_paths_from_nodes(
    create_edge_dict_graph: Dict[FrozenSet[int], Dict[str, Any]]
) -> None:
    """Test get_all_paths_from_nodes function.

    Args:
        create_edge_dict_graph (Dict[FrozenSet[int], Dict[str, Any]]): A fixture graph with 4 nodes and 6 edges.
    """
    g = defaultdict(list)
    for u, v in create_edge_dict_graph.keys():
        g[u].append(v)
        g[v].append(u)
    paths = get_all_paths_from_nodes([0, 1], g, 2)
    assert paths == set(
        [
            frozenset([0, 1]),
            frozenset([0, 2]),
            frozenset([0, 3]),
            frozenset([1, 2]),
            frozenset([1, 3]),
        ]
    )


def test_path_based_lift_CC(
    create_edge_dict_graph: Dict[FrozenSet[int], Dict[str, Any]]
) -> None:
    """
    Test path_based_lift_CC function.

    Args:
        create_edge_dict_graph (Dict[FrozenSet[int], Dict[str, Any]]): A fixture graph with 4 nodes and 6 edges.
    """
    cc = CombinatorialComplex()
    for cell in create_edge_dict_graph.keys():
        cc.add_cell(cell, rank=1, **create_edge_dict_graph[cell])

    res_cc = path_based_lift_CC(cc, [0, 1], 2)
    assert set(res_cc.cells.hyperedge_dict[2].keys()) == set(
        [
            frozenset([0, 1]),
            frozenset([0, 2]),
            frozenset([0, 3]),
            frozenset([1, 2]),
            frozenset([1, 3]),
        ]
    )

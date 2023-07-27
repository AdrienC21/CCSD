#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_cc_utils.py: test functions for cc_utils.py
"""

from collections import defaultdict
from typing import Tuple, List

import pytest
import torch
import numpy as np
import networkx as nx
from rdkit import Chem
from easydict import EasyDict
from toponetx.classes.combinatorial_complex import CombinatorialComplex

from src.utils.mol_utils import mols_to_nx
from src.utils.graph_utils import pad_adjs
from src.utils.cc_utils import (
    get_cells,
    cc_from_incidence,
    create_incidence_1_2,
    get_rank2_dim,
    get_mol_from_x_adj,
    get_all_mol_rings,
    mols_to_cc,
    CC_to_incidence_matrices,
    ccs_to_mol,
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
    rank2 = torch.tensor([[[1, 1, 1, 1] for _ in range(6)]], dtype=torch.float32)
    flags = torch.tensor([[1, 1, 0, 1]], dtype=torch.float32)  # remove node 2
    masked = mask_rank2(rank2, N, d_min, d_max, flags)
    assert masked.shape == (1, 6, 4)
    assert (
        (
            masked
            == torch.tensor(
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
            )
        )
        .all()
        .item()
    )


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
    assert (
        (
            As
            == torch.tensor(
                [
                    [0.0, 1.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        )
        .all()
        .item()
    )
    assert (
        (
            Fs
            == torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        )
        .all()
        .item()
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


def test_hodge_laplacian(create_incidence_1_2_test) -> None:
    """Test the hodge_laplacian function."""
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

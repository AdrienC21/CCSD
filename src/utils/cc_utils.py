#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""cc_utils.py: utility functions for combinatorial complex data (flag masking, conversions, etc.).
"""

import concurrent.futures
from typing import List, Tuple, Dict, FrozenSet, Optional, Union, Any, Callable
from itertools import combinations
from collections import defaultdict
from math import comb

import torch
import numpy as np
import networkx as nx
from rdkit import Chem
from easydict import EasyDict
from datetime import datetime
from toponetx.classes.combinatorial_complex import CombinatorialComplex

from src.utils.graph_utils import pad_adjs, node_flags, graphs_to_tensor
from src.utils.mol_utils import bond_decoder, SYMBOL_TO_AN, AN_TO_SYMBOL
from src.evaluation.stats import PRINT_TIME
from src.evaluation.mmd import compute_mmd, gaussian_emd, gaussian, gaussian_tv


DIC_MOL_CONV = {0: "C", 1: "N", 2: "O", 3: "F"}


def get_cells(
    N: int, d_min: int, d_max: int
) -> Tuple[
    List[FrozenSet[int]],
    Dict[FrozenSet[int], int],
    Dict[int, List[int]],
    List[FrozenSet[int]],
    Dict[FrozenSet[int], int],
    Dict[int, List[int]],
]:
    """Get all rank-2 cells of size d_min to d_max.
    Returns a list of all rank-2 cells, a dictionary mapping rank-2 cells to a
    column index in the incidence matrix, a dictionary mapping nodes to a list
    of column indices in the incidence matrix, a list of all edges,
    a dictionary mapping edges to a row index in the incidence matrix and a
    dictionary mapping nodes to a list of row indices in the incidence matrix.

    Args:
        N (int): maximum number of nodes
        d_min (int, optional): minimum size of rank-2 cells.
        d_max (int, optional): maximum size of rank-2 cells.

    Returns:
        Tuple[List[FrozenSet[int]], Dict[FrozenSet[int], int], Dict[int, List[int]], List[FrozenSet[int]], Dict[FrozenSet[int], int], Dict[int, List[int]]]: list of all rank-2 cells, dictionary mapping rank-2 cells to a column index in the incidence matrix, dictionary mapping nodes to a list of column indices in the incidence matrix, dictionary mapping edges to a row index in the incidence matrix and a dictionary mapping nodes to a list of row indices in the incidence matrix
    """

    # Get all the combinations of rank2 cells
    all_combinations = []
    nodes = list(range(N))
    for k in range(d_min, d_max + 1):
        all_combinations.extend(list(combinations(nodes, k)))
    all_combinations = [frozenset(c) for c in all_combinations]

    # Map all rank-2 cells to a column index in the incidence matrix and all nodes to a list of column indices
    dic_set = {}
    dic_int = defaultdict(list)
    for i, combi in enumerate(all_combinations):
        dic_set[combi] = i
        for n in combi:
            dic_int[n].append(i)
    # Map all edges to a row index in the incidence matrix and all nodes to a list of row indices
    # And get all the combinations of edges
    all_edges = []
    dic_edge = {}
    dic_int_edge = defaultdict(list)
    for i, edge in enumerate(list(combinations(nodes, 2))):
        all_edges.append(frozenset(edge))
        dic_edge[frozenset(edge)] = i
        dic_int_edge[edge[0]].append(i)
        dic_int_edge[edge[1]].append(i)

    return all_combinations, dic_set, dic_int, all_edges, dic_edge, dic_int_edge


def create_incidence_1_2(
    N: int,
    A: Union[np.ndarray, torch.Tensor],
    d_min: int,
    d_max: int,
    two_rank_cells: Dict[FrozenSet[int], Dict[str, Any]],
) -> np.ndarray:
    """Create the incidence matrix of rank-1 to rank-2 cells from an adjacency matrix
    and a list of the rank-2 cells of the CC.

    Args:
        N (int): maximum number of nodes
        A (Union[np.ndarray, torch.Tensor]): adjacency matrix
        d_min (int): minimum size of rank-2 cells
        d_max (int): maximum size of rank-2 cells
        two_rank_cells (Dict[FrozenSet[int], Dict[str, Any]]): list of rank-2 cells

    Returns:
        np.ndarray: incidence matrix of rank-1 to rank-2 cells
    """

    # Get all the combinations of nodes and the mapings
    all_combinations, dic_set, _, _, dic_edge, _ = get_cells(N, d_min, d_max)
    row = (N * (N - 1)) // 2
    col = len(all_combinations)
    if not (two_rank_cells):
        f = 1
    else:
        attributes_names = list(two_rank_cells[list(two_rank_cells.keys())[0]].keys())
        if "weight" in attributes_names:
            attributes_names.remove("weight")
        f = max(1, len(attributes_names))
    F = np.zeros((row, col, f), dtype=np.float32)  # empty incidence matrix

    if two_rank_cells:
        for c in two_rank_cells:
            j = dic_set[c]  # get the column index of the rank-2 cell
            combi = tuple(c)
            # For each pair of nodes in the rank-2 cell, get the row index of the edge
            for k in range(len(combi) - 1):
                for l in range(k + 1, len(combi)):
                    if (
                        A[combi[k], combi[l]].any() or A[combi[l], combi[k]].any()
                    ):  # if the edge exists
                        edge = frozenset((combi[k], combi[l]))
                        i = dic_edge[edge]
                        if not (attributes_names):
                            F[i, j, 0] = 1.0
                        else:
                            for attr_id, attr in enumerate(attributes_names):
                                F[i, j, attr_id] = two_rank_cells[c][attr]
    # Remove last dimension if only one attribute for rank2 incidence matrix
    if F.shape[-1] == 1:
        F = F.squeeze(-1)
    return F


def cc_from_incidence(
    incidence_matrices: Optional[
        Union[List[Optional[np.ndarray]], List[Optional[torch.Tensor]]]
    ],
    d_min: int,
    d_max: int,
    is_molecule: bool = False,
) -> CombinatorialComplex:
    """Convert (pseudo)-incidence matrices to a combinatorial complex (CC).

    Args:
        incidence_matrices (Optional[Union[List[Optional[np.ndarray]], List[Optional[torch.Tensor]]]]): list of incidence matrices [X, A, F]
        d_min (int, optional): minimum size of rank-2 cells.
        d_max (int, optional): maximum size of rank-2 cells.
        is_molecule (bool, optional): whether the CC is a molecule. Defaults to False.

    Raises:
        NotImplementedError: raise an error if the CC is of dimension greater than 2 (if len(incidence_matrices) > 3)

    Returns:
        CombinatorialComplex: combinatorial complex (CC) object
    """

    CC = CombinatorialComplex()
    # Empty CC. No incidence matrices, return empty CC
    if (
        (incidence_matrices is None)
        or (len(incidence_matrices) == 0)
        or (all(m is None for m in incidence_matrices))
    ):
        return CC

    # Convert to tensors
    incidence_matrices = [torch.Tensor(m) for m in incidence_matrices if m is not None]

    # 0-dimension CC. If only one incidence matrix, return CC with just nodes
    N = incidence_matrices[0].shape[0]
    for i in range(N):
        if incidence_matrices[0][i, :].any().item():
            if not (is_molecule):
                attr = {
                    f"label_{j}": incidence_matrices[0][i, j].item()
                    for j in range(incidence_matrices[0].shape[1])
                }
            else:
                attr = {
                    "symbol": SYMBOL_TO_AN[
                        DIC_MOL_CONV[torch.argmax(incidence_matrices[0][i, :]).item()]
                    ]
                }
            CC.add_cell((i,), rank=0, **attr)
    if len(incidence_matrices) == 1:
        return CC

    # 1-dimension CC. Two incidence matrices, return CC with nodes and edges
    adj_has_many_features = (
        len(incidence_matrices[1].shape) > 2
    )  # check if the adjacency matrix has many features
    for i in range(N):
        for j in range(i + 1, N):
            if incidence_matrices[1][i, j].any().item():
                if not (is_molecule):
                    if not (adj_has_many_features):
                        attr = {"label": incidence_matrices[1][i, j].item()}
                    else:
                        attr = {
                            f"label_{k}": incidence_matrices[1][i, j, k].item()
                            for k in range(incidence_matrices[1].shape[2])
                        }
                else:
                    if not (adj_has_many_features):
                        attr = {"bond_type": incidence_matrices[1][i, j].item()}
                    else:
                        bond_type = torch.argmax(incidence_matrices[1][i, j]).item()
                        attr = {"bond_type": bond_type}
                CC.add_cell((i, j), rank=1, **attr)
    if len(incidence_matrices) == 2:
        return CC

    # 2-dimension CC. If three incidence matrices, return CC with nodes, edges and rank-2 cells
    incidence_matrix = incidence_matrices[2]
    rank2_has_many_features = (
        len(incidence_matrix.shape) > 2
    )  # check if the rank2 incidence matrix has many features
    all_combinations, _, _, _, _, _ = get_cells(N, d_min, d_max)
    for i, combi in enumerate(all_combinations):
        if incidence_matrix[:, i].any().item():  # there is a rank2 cell
            label_index = incidence_matrix[:, i].abs().argmax().item()
            if not (rank2_has_many_features):
                attr = {"label": incidence_matrix[label_index, i].item()}
            else:
                label_index = label_index // incidence_matrix.shape[2]
                attr = {
                    f"label_{k}": incidence_matrix[label_index, i, k].item()
                    for k in range(incidence_matrix.shape[2])
                }
            CC.add_cell(combi, 2, **attr)
    if len(incidence_matrices) == 3:
        return CC
    else:  # if more than 3 incidence matrices, return an error
        raise NotImplementedError(
            "Combinatorial Complexes of dimension > 2 not implemented"
        )


def get_rank2_dim(N: int, d_min: int, d_max: int) -> int:
    """Get the dimension of the rank-2 incidence matrix of a combinatorial complex
    with the given parameters.

    Args:
        N (int): maximum number of nodes
        d_min (int): minimum size of rank-2 cells
        d_max (int): maximum size of rank-2 cells

    Returns:
        int: dimension of the rank-2 incidence matrix
    """
    rows = (N * (N - 1)) // 2
    cols = sum([comb(N, i) for i in range(d_min, d_max + 1)])
    return rows, cols


def get_mol_from_x_adj(x: torch.Tensor, adj: torch.Tensor) -> Chem.Mol:
    """Get a molecule from the node and adjacency matrices after
    being processed by get_transform_fn inside data_loader_mol.py.

    Atoms:
    0: C, 1: N, 2: O, 3: F
    Bonds:
    1: single, 2: double, 3: triple

    Args:
        x (torch.Tensor): node matrix
        adj (torch.Tensor): adjacency matrix

    Returns:
        Chem.Mol: molecule (RDKIT mol)
    """
    mol = Chem.RWMol()
    for i in range(x.shape[0]):
        if x[i].any():
            atom_symbol = DIC_MOL_CONV[torch.argmax(x[i]).item()]
            mol.AddAtom(Chem.Atom(atom_symbol))

    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j]:
                mol.AddBond(i, j, bond_decoder[adj[i, j].item()])

    mol = mol.GetMol()  # convert from RWMol (editable) to Mol
    return mol


def get_all_mol_rings(mol: Chem.Mol) -> List[FrozenSet[int]]:
    """Get all the rings of a molecule.

    Args:
        mol (Chem.Mol): molecule (RDKIT mol)

    Returns:
        List[FrozenSet[int]]: list of rings as frozensets of atom indices
    """
    res = []
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        ring_list = []
        for atom in ring:
            ring_list.append(atom)
        res.append(frozenset(sorted(ring_list)))
    return res


def mols_to_cc(mols: List[Chem.Mol]) -> List[CombinatorialComplex]:
    """Convert a list of molecules to a list of combinatorial complexes
    where the rings are rank-2 cells.

    This is a general function mostly used for testing.
    A more complete one is implemented in src/utils/data_loader_mol.py
    within the MolDataset class.

    Args:
        mols (List[Chem.Mol]): list of molecules (RDKIT mol)

    Returns:
        List[CombinatorialComplex]: molecules as combinatorial complexes
        where the cycles are rank-2 cells

    Example:
        >>> mols = [Chem.MolFromSmiles("Cc1ccccc1"), Chem.MolFromSmiles("c1cccc2c1CCCC2")]
        >>> ccs = mols_to_cc(mols)
    """
    ccs = []
    for mol in mols:
        CC = CombinatorialComplex()

        # Atom
        for atom in mol.GetAtoms():
            CC.add_cell((atom.GetIdx(),), rank=0, symbol=SYMBOL_TO_AN[atom.GetSymbol()])

        # Bond
        for bond in mol.GetBonds():
            CC.add_cell(
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                rank=1,
                bond_type=bond.GetBondTypeAsDouble(),
            )

        # Ring as rank-2 cells
        for rings in get_all_mol_rings(mol):
            CC.add_cell(rings, rank=2)

        ccs.append(CC)
    return ccs


def CC_to_incidence_matrices(
    CC: CombinatorialComplex, d_min: Optional[int], d_max: Optional[int]
) -> List[np.ndarray]:
    """Convert a combinatorial complex to a list of incidence matrices.

    Args:
        CC (CombinatorialComplex): combinatorial complex
        d_min (Optional[int]): minimum size of rank-2 cells. If not provided, calculated from the CC
        d_max (Optional[int]): maximum size of rank-2 cells. If not provided, calculated from the CC

    Returns:
        List[np.ndarray]: list of incidence matrices
    """

    if not (CC.cells.hyperedge_dict):  # empty CC
        return [np.array([]), np.array([]), np.array([])]

    # Nodes
    nodes = CC.cells.hyperedge_dict[0]
    N = len(nodes)
    if not (nodes):
        f = 1
    else:
        attributes_names = list(nodes[list(nodes.keys())[0]].keys())
        if "weight" in attributes_names:
            attributes_names.remove("weight")
        f = max(1, len(attributes_names))
    X = np.zeros((N, f), dtype=np.float32)
    if nodes:
        for k in list(nodes.keys()):
            node = tuple(k)[0]
            if not (attributes_names):
                X[node, 0] = 1
            else:
                for attr_id, attr in enumerate(attributes_names):
                    X[node, attr_id] = nodes[k][attr]

    # Edges
    if 1 not in CC.cells.hyperedge_dict:
        return [X, np.array([]), np.array([])]
    edges = CC.cells.hyperedge_dict[1]
    if not (edges):
        f = 1
    else:
        attributes_names = list(edges[list(edges.keys())[0]].keys())
        if "weight" in attributes_names:
            attributes_names.remove("weight")
        f = max(1, len(attributes_names))
    A = np.zeros((N, N, f), dtype=np.float32)
    if edges:
        for k in list(edges.keys()):
            edge = tuple(k)
            u, v = edge[0], edge[1]
            if not (attributes_names):
                A[u, v, 0] = 1.0
                A[v, u, 0] = 1.0
            else:
                for attr_id, attr in enumerate(attributes_names):
                    A[u, v, attr_id] = edges[k][attr]
                    A[v, u, attr_id] = edges[k][attr]
    # Remove last dimension if only one attribute for adjacency matrix
    if A.shape[-1] == 1:
        A = A.squeeze(-1)

    # Rank-2 cells
    if 2 not in CC.cells.hyperedge_dict:
        return [X, A, np.array([])]
    rank_2_cells = CC.cells.hyperedge_dict[2]
    d_min = min(len(c) for c in rank_2_cells) if d_min is None else d_min
    d_max = min(len(c) for c in rank_2_cells) if d_max is None else d_max
    F = create_incidence_1_2(N, A, d_min, d_max, rank_2_cells)
    return [X, A, F]


def ccs_to_mol(ccs: List[CombinatorialComplex]) -> List[Chem.Mol]:
    """Convert a list of combinatorial complexes to a list of molecules.

    Args:
        ccs (List[CombinatorialComplex]): list of combinatorial complexes
        that represent molecules to convert

    Returns:
        List[Chem.Mol]: list of molecules
    """
    mols = []
    for cc in ccs:
        mol = Chem.RWMol()

        atoms = cc.cells.hyperedge_dict[0]
        for atom in atoms:
            atom_symbol = AN_TO_SYMBOL[atoms[atom]["symbol"]]
            mol.AddAtom(Chem.Atom(atom_symbol))

        bonds = cc.cells.hyperedge_dict[1]
        for b in bonds:
            bond = tuple(b)
            atom_a = bond[0]
            atom_b = bond[1]
            bond_type = bond_decoder[bonds[b]["bond_type"]]
            mol.AddBond(atom_a, atom_b, bond_type)

        mol = mol.GetMol()  # convert from RWMol (editable) to Mol
        mols.append(mol)

    return mols


def get_N_from_rank2(rank2: torch.Tensor) -> int:
    """Get number of nodes from batch of rank2 incidence matrices

    Args:
        rank2 (torch.Tensor): rank2 incidence matrices (raw, batch, or batch and channel).
            (NC2) x K or B x (NC2) x K or B x C x (NC2) x K

    Returns:
        int: number of nodes
    """
    if len(rank2.shape) == 2:  # no batch
        nb_edges = rank2.shape[0]
    elif len(rank2.shape) == 4:  # batch and channel
        nb_edges = rank2.shape[2]
    else:
        nb_edges = rank2.shape[1]
    N = int((1 + np.sqrt(1 + 8 * nb_edges)) / 2)
    return N


def get_rank2_flags(
    rank2: torch.Tensor, N: int, d_min: int, d_max: int, flags: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get flags for left and right nodes of rank2 cells.
    The left flag is 0 if the edge is not in the CC as a node is not.
    The right flag is 0 if the rank-2 cell is not in the CC as a node is not.

    Args:
        rank2 (torch.Tensor): batch of rank2 incidence matrices.
            B x (NC2) x K or B x C x (NC2) x K
        N (int): number of nodes
        d_min (int): minimum dimension of rank2 cells
        d_max (int): maximum dimension of rank2 cells
        flags (torch.Tensor): 0-1 flags tensor. B x N

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: flags for left and right nodes of rank2 cells
    """
    _, _, dic_int, _, _, dic_int_edge = get_cells(N, d_min, d_max)
    nb_edges, K = rank2.shape[-2:]
    flags_left = torch.ones((rank2.shape[0], nb_edges), device=rank2.device)
    flags_right = torch.ones((rank2.shape[0], K), device=rank2.device)
    for b in range(flags.shape[0]):
        for n in range(flags.shape[1]):
            if not (flags[b, n]):  # node n is not in the CC
                for i in dic_int_edge[n]:  # remove the flags of the edges containing n
                    flags_left[b, i] = 0
                for j in dic_int[n]:  # remove the flags of the rank2 cells containing n
                    flags_right[b, j] = 0
    return flags_left, flags_right


def mask_rank2(
    rank2: torch.Tensor,
    N: int,
    d_min: int,
    d_max: int,
    flags: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mask batch of rank2 incidence matrices with 0-1 flags tensor

    Args:
        rank2 (torch.Tensor): batch of rank2 incidence matrices.
            B x (NC2) x K or B x C x (NC2) x K
        N (int): number of nodes
        d_min (int): minimum dimension of rank2 cells
        d_max (int): maximum number of rank2 cells
        flags (Optional[torch.Tensor], optional): 0-1 flags tensor. Defaults to None.
            B x N

    Returns:
        torch.Tensor: Mask batch of rank2 incidence matrices
    """
    if flags is None:
        flags = torch.ones((rank2.shape[0], N), device=rank2.device)

    flags_left, flags_right = get_rank2_flags(rank2, N, d_min, d_max, flags)

    if len(rank2.shape) == 4:
        flags_left = flags_left.unsqueeze(1)  # B x 1 x (NC2)
        flags_right = flags_right.unsqueeze(1)  # B x 1 x K

    rank2 = flags_left.unsqueeze(-1) * rank2 * flags_right.unsqueeze(-2)
    return rank2


def gen_noise_rank2(
    x: torch.Tensor,
    N: int,
    d_min: int,
    d_max: int,
    flags: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate noise for the rank-2 incidence matrix

    Args:
        x (torch.Tensor): input tensor
        N (int): number of nodes
        d_min (int): minimum dimension of rank2 cells
        d_max (int): maximum dimension of rank2 cells
        flags (Optional[torch.Tensor], optional): optional flags. Defaults to None.

    Returns:
        torch.Tensor: generated noisy tensor
    """
    z = torch.randn_like(x)  # gaussian centered normal distribution
    z = mask_rank2(z, N, d_min, d_max, flags)
    return z


def pad_rank2(
    ori_rank2: np.ndarray, node_number: int, d_min: int, d_max: int
) -> np.ndarray:
    """Create padded rank2 incidence matrices

    Args:
        ori_adj (np.ndarray): original rank2 incidence matrix
        node_number (int): number of desired nodes
        d_min (int): minimum dimension of rank2 cells
        d_max (int): maximum dimension of rank2 cells

    Raises:
        ValueError: if the original rank2 incidence matrix has more nodes larger than the desired number of nodes (we can't pad)

    Returns:
        np.ndarray: Padded adjacency matrix
    """
    if not (ori_rank2.size):
        rows, cols = get_rank2_dim(node_number, d_min, d_max)
        return np.zeros((rows, cols), dtype=np.float32)
    r = ori_rank2
    ori_len = get_N_from_rank2(r)
    if ori_len == node_number:  # same shape
        return r
    if ori_len > node_number:
        raise ValueError(
            f"Original number of nodes {ori_len} is greater (>) that the desired number of nodes after padding {node_number}"
        )
    # Pad
    all_combinations, _, _, all_edges, _, _ = get_cells(ori_len, d_min, d_max)
    new_all_combinations, new_dic_set, _, new_all_edges, new_dic_edge, _ = get_cells(
        node_number, d_min, d_max
    )
    res = np.zeros([len(new_all_edges), len(new_all_combinations)])
    for i, edge in enumerate(all_edges):
        for j, comb in enumerate(all_combinations):
            new_i = new_dic_edge[edge]
            new_j = new_dic_set[comb]
            res[new_i, new_j] = r[i, j]
    return res


def get_global_cc_properties(ccs: List[CombinatorialComplex]) -> Tuple[int, int, int]:
    """Get the global properties of a list of combinatorial complexes:
    number of nodes, minimum dimension of rank2 cells and maximum dimension of rank2 cells

    Args:
        ccs (List[CombinatorialComplex]): list of combinatorial complexes

    Returns:
        Tuple[int, int, int]: number of nodes, minimum dimension of rank2 cells and maximum dimension of rank2 cells

    Example:
        >>> mols = [Chem.MolFromSmiles("Cc1ccccc1"), Chem.MolFromSmiles("c1cccc2c1CCCC2"), Chem.MolFromSmiles("C1CC1")]
        >>> ccs = mols_to_cc(mols)
        >>> get_global_cc_properties(ccs)
        (10, 3, 6)
    """
    max_node_num = max(len(cc.cells.hyperedge_dict.get(0, [])) for cc in ccs)
    d_min = min(
        min(len(cell) for cell in cc.cells.hyperedge_dict.get(2, [])) for cc in ccs
    )
    d_max = max(
        max(len(cell) for cell in cc.cells.hyperedge_dict.get(2, [])) for cc in ccs
    )
    return max_node_num, d_min, d_max


def ccs_to_tensors(
    cc_list: List[CombinatorialComplex],
    max_node_num: Optional[int] = None,
    d_min: Optional[int] = None,
    d_max: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a list of combinatorial complexes to two tensors, one for the adjacency matrices and one for the incidence matrices
    If the combinatorial complexes have different number of nodes, the adjacency matrices and incidence matrices
    are padded to the maximum number of nodes.
    If the max number of nodes is not provided, it is calculated from the combinatorial complexes.
    Same for the minimum and maximum dimension of rank2 cells.

    Args:
        cc_list (List[CombinatorialComplex]): list of combinatorial complexes
        max_node_num (Optional[int], optional): max number of nodes in all the combinatorial complexes. Defaults to None.
        d_min (Optional[int], optional): minimum dimension of rank2 cells. Defaults to None.
        d_max (Optional[int], optional): maximum dimension of rank2 cells. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: adjacency matrices and rank2 incidence matrices
    """
    adjs_list = []
    rank2_list = []
    max_node_num = max_node_num  # memory issue

    if (max_node_num is None) or (d_min is None) or (d_max is None):
        max_node_num, d_min, d_max = get_global_cc_properties(cc_list)

    for cc in cc_list:
        assert isinstance(cc, CombinatorialComplex)

        _, adj, rank2 = CC_to_incidence_matrices(cc, d_min, d_max)

        # convert to adj matrix
        padded_adj = pad_adjs(adj, node_number=max_node_num)  # pad to max node number
        adjs_list.append(padded_adj)

        # convert to rank2 incidence matrix
        padded_rank2 = pad_rank2(
            rank2, node_number=max_node_num, d_min=d_min, d_max=d_max
        )  # pad to max node number
        rank2_list.append(padded_rank2)

    del cc_list

    adjs_np = np.asarray(adjs_list)  # concatenate the arrays
    rank2_np = np.asarray(rank2_list)
    del adjs_list
    del rank2_list

    adjs_tensor = torch.tensor(adjs_np, dtype=torch.float32)  # convert to tensor
    rank2_tensor = torch.tensor(rank2_np, dtype=torch.float32)
    del adjs_np
    del rank2_np

    return adjs_tensor, rank2_tensor


def cc_to_tensor(
    cc: CombinatorialComplex,
    max_node_num: Optional[int] = None,
    d_min: Optional[int] = None,
    d_max: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a single combinatorial complex to a tuple of tensors, one for the adjacency matrix and one for the rank2 incidence matrix
    If the max number of nodes is not provided, it is calculated from the combinatorial complexes.
    Same for the minimum and maximum dimension of rank2 cells.
    Incidence matrices (A, F) are padded to the maximum number of nodes.

    Args:
        cc (CombinatorialComplex): combinatorial complex to convert
        max_node_num (Optional[int], optional): maximum number of nodes. Defaults to None.
        d_min (Optional[int], optional): minimum dimension of rank2 cells. Defaults to None.
        d_max (Optional[int], optional): maximum dimension of rank2 cells. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: adjacency matrix and rank2 incidence matrix
    """
    max_node_num = max_node_num  # memory issue

    if (max_node_num is None) or (d_min is None) or (d_max is None):
        max_node_num, d_min, d_max = get_global_cc_properties([cc])

    assert isinstance(cc, CombinatorialComplex)
    _, adj, rank2 = CC_to_incidence_matrices(cc, d_min, d_max)

    # convert to adj matrix
    padded_adj = pad_adjs(adj, node_number=max_node_num)  # pad to max node number

    # convert to rank2 incidence matrix
    padded_rank2 = pad_rank2(
        rank2, node_number=max_node_num, d_min=d_min, d_max=d_max
    )  # pad to max node number

    adj = torch.tensor(padded_adj, dtype=torch.float32)
    rank2 = torch.tensor(padded_rank2, dtype=torch.float32)
    del padded_adj
    del padded_rank2

    return adj, rank2


def convert_CC_to_graphs(
    ccs: List[CombinatorialComplex], undirected: bool = True
) -> List[nx.Graph]:
    """Convert a list of combinatorial complexes to a list of graphs

    Args:
        ccs (List[CombinatorialComplex]): list of combinatorial complexes
        undirected (bool, optional): whether to create an undirected graph. Defaults to True.

    Returns:
        List[nx.Graph]: list of graphs
    """
    graphs = []
    for cc in ccs:
        graph = nx.Graph()
        for node in cc.cells.hyperedge_dict[0]:
            n = tuple(node)[0]
            graph.add_node(n, **cc.cells.hyperedge_dict[0][node])
        for edge in cc.cells.hyperedge_dict[1]:
            u = tuple(edge)[0]
            v = tuple(edge)[1]
            graph.add_edge(u, v, **cc.cells.hyperedge_dict[1][edge])
            if undirected:
                graph.add_edge(v, u, **cc.cells.hyperedge_dict[1][edge])
        graphs.append(graph)
    return graphs


def convert_graphs_to_CCs(
    graphs: List[nx.Graph], is_molecule: bool = False
) -> List[CombinatorialComplex]:
    """Convert a list of graphs to a list of combinatorial complexes (of dimension 1).

    Args:
        graphs (List[nx.Graph]): list of graphs
        is_molecule (bool, optional): whether the graphs are molecules. Defaults to False.

    Returns:
        List[CombinatorialComplex]: list of combinatorial complexes
    """
    ccs = []
    for graph in graphs:
        CC = CombinatorialComplex()
        for node in graph.nodes:
            attr = graph.nodes[node]
            if is_molecule and isinstance(attr["label"], str):
                attr["symbol"] = attr["label"]
                del attr["label"]
                attr["symbol"] = SYMBOL_TO_AN[attr["symbol"]]
            CC.add_cell((node,), rank=0, **attr)
        for edge in graph.edges:
            attr = graph.edges[edge]
            if is_molecule:
                attr["bond_type"] = float(attr["label"])
                del attr["label"]
            CC.add_cell(edge, rank=1, **attr)
        ccs.append(CC)
    return ccs


def init_flags(
    obj_list: Union[List[nx.Graph], List[CombinatorialComplex]],
    config: EasyDict,
    batch_size: Optional[int] = None,
    is_cc: bool = False,
) -> torch.Tensor:
    """Sample initial flags tensor from the training graph set

    Args:
        graph_list (List[nx.Graph]): list of graphs
        config (EasyDict): configuration
        batch_size (Optional[int], optional): batch size. Defaults to None.
        is_cc (bool, optional): is the objects combinatorial complexes?. Defaults to False.

    Returns:
        torch.Tensor: flag tensors
    """

    if batch_size is None:  # get a default one from the config
        batch_size = config.data.batch_size
    max_node_num = config.data.max_node_num
    if not (is_cc):
        graph_tensor = graphs_to_tensor(obj_list, max_node_num)
        idx = np.random.randint(0, len(obj_list), batch_size)
        flags = node_flags(graph_tensor[idx])
    else:
        d_min = config.data.d_min
        d_max = config.data.d_max
        cc_tensor = ccs_to_tensors(obj_list, max_node_num, d_min, d_max)
        idx = np.random.randint(0, len(obj_list), batch_size)
        flags = node_flags(cc_tensor[0][idx])
    return flags


def hodge_laplacian(rank2: torch.Tensor) -> torch.Tensor:
    """Compute the Hodge Laplacian of a batch of rank2 incidence matrices.
    H = F @ F.T where F is the rank-2 incidence matrix of a combinatorial complex.

    Args:
        rank2 (torch.Tensor): batch of rank2 incidence matrices.
            B x (NC2) x K or B x C x (NC2) x K

    Returns:
        torch.Tensor: Hodge Laplacian
            B x (NC2) x (NC2) or B x C x (NC2) x (NC2)
    """
    return rank2 @ rank2.transpose(-1, -2)


def default_mask(n: int) -> torch.Tensor:
    """Create default adjacency or Hodge Laplacian mask (no diagonal elements)

    Args:
        n (int): number of nodes or edges

    Returns:
        torch.Tensor: default adjacency or Hodge Laplacian mask
    """
    return torch.ones([n, n]) - torch.eye(n)


def pow_tensor_cc(
    x: torch.Tensor, cnum: int, hodge_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Create higher order rank-2 incidence matrices from a batch of rank-2 incidence matrices.

    Args:
        x (torch.Tensor): input tensor of shape B x (NC2) x K
        cnum (int): number of higher order matrices to create
            (made with consecutive multiplication of the Hodge Laplacian matrix of x)
        hodge_mask (Optional[torch.Tensor], optional): optional mask to apply to the Hodge Laplacian.
            Defaults to None. If None, no mask is applied.
            shape (NC2) x (NC2) or B x (NC2) x (NC2)

    Returns:
        torch.Tensor: output higher order matrices of shape B x cnum x (NC2) x K
    """
    x_ = x.clone()
    H = hodge_laplacian(x)
    if hodge_mask is not None:
        if len(hodge_mask.shape) == 2:  # make it batched
            hodge_mask = hodge_mask.unsqueeze(0)
    H = H * hodge_mask if hodge_mask is not None else H
    xc = [x.unsqueeze(1)]
    for _ in range(cnum - 1):
        x_ = torch.bmm(H, x_)
        xc.append(x_.unsqueeze(1))
    xc = torch.cat(xc, dim=1)

    return xc


def is_empty_cc(cc: CombinatorialComplex) -> bool:
    """Check if a combinatorial complex is empty

    Args:
        cc (CombinatorialComplex): combinatorial complex

    Returns:
        bool: whether the combinatorial complex is empty
    """
    return cc.number_of_cells() == 0


def rank2_distrib_worker(
    CC: CombinatorialComplex, d_min: int, d_max: int
) -> np.ndarray:
    """Function for computing the rank-2 cell histogram of a combinatorial complex.

    Returns:
        np.ndarray: rank-2 cell histogram
        d_min (int): minimum dimension of the rank-2 cells
        d_max (int): maximum dimension of the rank-2 cells
    """
    rank2_cells = CC.cells.hyperedge_dict.get(2, {})
    rank2_distrib = np.zeros(d_max - d_min + 1)
    for cell in rank2_cells:
        length = len(cell)
        if (d_min <= length) and (length <= d_max):
            rank2_distrib[length - d_min] += 1
    return rank2_distrib


def rank2_distrib_stats(
    cc_ref_list: List[CombinatorialComplex],
    cc_pred_list: List[CombinatorialComplex],
    d_min: int,
    d_max: int,
    kernel: Callable[[np.ndarray, np.ndarray], float] = gaussian_emd,
    is_parallel: bool = True,
) -> float:
    """Compute the MMD distance between the nummber of rank-2 cells distributions of two unordered sets of combinatorial complexes.

    Args:
        cc_ref_list (List[CombinatorialComplex]): reference list of toponetx combinatorial complexes to be evaluated
        cc_pred_list (List[CombinatorialComplex]): target list of toponetx combinatorial complexes to be evaluated
        d_min (int): minimum dimension of the rank-2 cells
        d_max (int): maximum dimension of the rank-2 cells
        kernel (Callable[[np.ndarray, np.ndarray], float], optional): kernel function. Defaults to gaussian_emd.
        is_parallel (bool, optional): if True, do parallel computing. Defaults to True.

    Returns:
        float: MMD distance
    """

    sample_ref = []
    sample_pred = []
    # Remove empty CCs if generated
    cc_pred_list_remove_empty = [cc for cc in cc_pred_list if not is_empty_cc(cc)]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for rank2_distrib_hist in executor.map(
                lambda cc: rank2_distrib_worker(cc, d_min, d_max), cc_ref_list
            ):
                sample_ref.append(rank2_distrib_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for rank2_distrib_hist in executor.map(
                lambda cc: rank2_distrib_worker(cc, d_min, d_max),
                cc_pred_list_remove_empty,
            ):
                sample_pred.append(rank2_distrib_hist)

    else:
        for i in range(len(cc_ref_list)):
            rank2_distrib_temp = rank2_distrib_worker(cc_ref_list[i], d_min, d_max)
            sample_ref.append(rank2_distrib_temp)
        for i in range(len(cc_pred_list_remove_empty)):
            rank2_distrib_temp = rank2_distrib_worker(
                cc_pred_list_remove_empty[i], d_min, d_max
            )
            sample_pred.append(rank2_distrib_temp)
    # Compute MMD
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=kernel)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


# Dictionary mapping method names to functions to compute different MMD distances
CC_METHOD_NAME_TO_FUNC = {
    "rank2_distrib": rank2_distrib_stats,
}


def eval_CC_list(
    cc_ref_list: List[CombinatorialComplex],
    cc_pred_list: List[CombinatorialComplex],
    d_min: int,
    d_max: int,
    methods: Optional[List[str]] = None,
    kernels: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
) -> Dict[str, float]:
    """Evaluate generated generic combinatorial complexes against a reference set of combinatorial complexes using a set of methods and their corresponding kernels.

    Args:
        cc_ref_list (List[CombinatorialComplex]): reference list of toponetx combinatorial complexes to be evaluated
        cc_pred_list (List[CombinatorialComplex]): target list of toponetx combinatorial complexes to be evaluated
        d_min (int): minimum dimension of the rank-2 cells
        d_max (int): maximum dimension of the rank-2 cells
        methods (Optional[List[str]], optional): methods to be evaluated. Defaults to None.
        kernels (Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]], optional): kernels to be used for each methods. Defaults to None.

    Returns:
        Dict[str, float]: dictionary mapping method names to their corresponding scores
    """
    if methods is None:  # by default, evaluate the methods ["rank2_distrib"]
        methods = ["rank2_distrib"]
    results = {}
    for method in methods:
        results[method] = round(
            CC_METHOD_NAME_TO_FUNC[method](
                cc_ref_list, cc_pred_list, d_min, d_max, kernels[method]
            ),
            6,
        )
        print(
            "\033[91m"
            + f"{method:9s}"
            + "\033[0m"
            + " : "
            + "\033[94m"
            + f"{results[method]:.6f}"
            + "\033[0m"
        )
    return results


def load_cc_eval_settings() -> (
    Tuple[List[str], Dict[str, Callable[[np.ndarray, np.ndarray], float]]]
):
    """Load the methods and kernels to be used for evaluating combinatorial complexes.

    Returns:
        Tuple[List[str], Dict[str, Callable[[np.ndarray, np.ndarray], float]]]: methods and kernels to be used for evaluating combinatorial complexes
    """
    # Methods to use (from [rank2_distrib], see utils/cc_utils.py)
    methods = ["rank2_distrib"]
    # Kernels to use for each method (from [gaussian, gaussian_emd, gaussian_tv], see evaluation/mmd.py)
    kernels = {
        "rank2_distrib": gaussian_emd,
    }
    return methods, kernels


def adj_to_hodgedual(adj: torch.Tensor) -> torch.Tensor:
    """Convert a batch and channels of adjacency matrices to a batch and channels of Hodge dual adjacency matrices.

    Args:
        adj (torch.Tensor): adjacency matrices (B x C x N x N)

    Returns:
        torch.Tensor: Hodge dual adjacency matrices (B x C x (NC2) x (NC2))
    """
    # Get shapes
    batch_size, channels, N, _ = adj.shape
    hodgedual_size = (N * (N - 1)) // 2
    # Extract diagonal coefficients that become diagonal coefficients of Hodge dual
    upper_triangle = torch.triu(adj, diagonal=1)
    diag = torch.masked_select(upper_triangle, upper_triangle != 0)
    # Reshape to (B x C x (NC2))
    diag = diag.reshape(batch_size, channels, hodgedual_size)
    # Convert to Hodge dual
    hodgedual = torch.diag_embed(diag)
    hodgedual = hodgedual.to(adj.device)
    return hodgedual


def hodgedual_to_adj(hodgedual: torch.Tensor) -> torch.Tensor:
    """Convert a batch and channels of Hodge dual adjacency matrices to a batch and channels of adjacency matrices.

    Args:
        hodgedual (torch.Tensor): Hodge dual adjacency matrices (B x C x (NC2) x (NC2))

    Returns:
        torch.Tensor: adjacency matrices (B x C x N x N)
    """
    # Get shapes
    batch_size, channels, hodgedual_size, _ = hodgedual.shape
    N = int(
        (1 + (1 + 8 * hodgedual_size) ** 0.5) / 2
    )  # solve (N*(N-1))/2 = hodgedual_size

    # Extract diagonal coefficients from Hodge dual along dimensions (NC2) x (NC2)
    diag = hodgedual.diagonal(dim1=2, dim2=3)

    # Reshape to (B x C x N x N)
    rows, cols = torch.tril_indices(N, N, offset=-1)  # indices of lower triangle
    # Sort to go from (0,1) ... (0,N-1), (1,2) etc.
    sorted_index = sorted(zip(rows, cols), key=lambda pair: (pair[1], pair[0]))
    rows, cols = zip(*sorted_index)
    # Convert to tensors
    rows, cols = torch.tensor(rows, device="cpu"), torch.tensor(cols, device="cpu")
    # Create adjacency matrices
    adj = torch.zeros(batch_size, channels, N, N, device=hodgedual.device)
    adj[:, :, rows, cols] = diag
    adj[:, :, cols, rows] = diag  # symmetrize the adjacency matrices (undirected)

    return adj

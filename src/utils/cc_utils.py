#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""cc_utils.py: utility functions for combinatorial complex data (flag masking, conversions, etc.).
"""

from typing import List, Tuple, Dict, FrozenSet, Optional, Union
from itertools import combinations
from collections import defaultdict
from math import comb

import torch
import numpy as np
import networkx as nx
from rdkit import Chem
from toponetx.classes.combinatorial_complex import CombinatorialComplex

from src.utils.graph_utils import pad_adjs
from src.utils.mol_utils import bond_decoder


DIC_MOL_CONV = {0: "C", 1: "N", 2: "O", 3: "F"}


def get_cells(
    N: int, d_min: int = 3, d_max: int = 9
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
        d_min (int, optional): minimum size of rank-2 cells. Defaults to 3.
        d_max (int, optional): maximum size of rank-2 cells. Defaults to 9.

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


def cc_from_incidence(
    incidence_matrices: Optional[
        Union[List[Optional[np.ndarray]], List[Optional[torch.Tensor]]]
    ],
    d_min: int = 3,
    d_max: int = 6,
    is_molecule: bool = False,
) -> CombinatorialComplex:
    """Convert (pseudo)-incidence matrices to a combinatorial complex (CC).

    Args:
        incidence_matrices (Optional[Union[List[Optional[np.ndarray]], List[Optional[torch.Tensor]]]]): list of incidence matrices [X, A, F]
        d_min (int, optional): minimum size of rank-2 cells. Defaults to 3.
        d_max (int, optional): maximum size of rank-2 cells. Defaults to 6.
        is_molecule (bool, optional): whether the CC is a molecule. Defaults to False.

    Raises:
        NotImplementedError: raise an error if the CC is of dimension greater than 2 (if len(incidence_matrices) > 3)

    Returns:
        CombinatorialComplex: combinatorial complex (CC) object
    """

    CC = CombinatorialComplex()
    # Empty CC. No incidence matrices, return empty CC
    if (incidence_matrices is None) or (len(incidence_matrices) == 0):
        return CC

    # Convert to tensors
    incidence_matrices = [torch.Tensor(m) for m in incidence_matrices]

    # 0-dimension CC. One incidence matrix, return CC with just nodes
    N = incidence_matrices[0].shape[0]
    if len(incidence_matrices) == 1:
        for i in range(N):
            if incidence_matrices[0][i, :].any().item():
                if not (is_molecule):
                    attr = {
                        f"label_{j}": incidence_matrices[0][i, j].item()
                        for j in range(incidence_matrices[0].shape[1])
                    }
                else:
                    attr = {
                        "symbol": DIC_MOL_CONV[
                            torch.argmax(incidence_matrices[0][i, :]).item()
                        ]
                    }
                CC.add_cell((i,), rank=0, **attr)
        return CC

    # 1-dimension CC. Two incidence matrices, return CC with nodes and edges
    for i in range(N):
        for j in range(i + 1, N):
            if incidence_matrices[1][i, j]:
                if not (is_molecule):
                    attr = {"label": incidence_matrices[1][i, j].item()}
                else:
                    attr = {"bond_type": incidence_matrices[1][i, j].item()}
                CC.add_cell((i, j), rank=1, **attr)
    if len(incidence_matrices) == 2:
        return CC

    # 2-dimension CC. Three incidence matrices, return CC with nodes, edges and rank-2 cells
    if len(incidence_matrices) == 3:
        incidence_matrix = incidence_matrices[2]
        all_combinations, _, _, _, _, _ = get_cells(N, d_min, d_max)
        for i, combi in enumerate(all_combinations):
            if incidence_matrix[:, i].any().item():
                CC.add_cell(combi, 2)
        return CC
    else:
        raise NotImplementedError()


def create_incidence_1_2(
    N: int, A: np.ndarray, d_min: int, d_max: int, two_rank_cells: List[FrozenSet[int]]
) -> np.ndarray:
    """Create the incidence matrix of rank-1 to rank-2 cells from an adjacency matrix
    and a list of the rank-2 cells of the CC.

    Args:
        N (int): maximum number of nodes
        A (np.ndarray): adjacency matrix
        d_min (int): minimum size of rank-2 cells
        d_max (int): maximum size of rank-2 cells
        two_rank_cells (List[FrozenSet[int]]): list of rank-2 cells

    Returns:
        np.ndarray: incidence matrix of rank-1 to rank-2 cells
    """

    # Get all the combinations of nodes and the mapings
    all_combinations, dic_set, _, _, dic_edge, _ = get_cells(N, d_min, d_max)
    row = (N * (N - 1)) // 2
    col = len(all_combinations)
    res = np.zeros((row, col))  # empty incidence matrix
    for c in two_rank_cells:
        j = dic_set[c]  # get the column index of the rank-2 cell
        combi = list(c)
        # For each pair of nodes in the rank-2 cell, get the row index of the edge
        for k in range(len(combi) - 1):
            for l in range(k + 1, len(combi)):
                if A[combi[k], combi[l]] or A[combi[l], combi[k]]:  # if the edge exists
                    edge = frozenset((combi[k], combi[l]))
                    i = dic_edge[edge]
                    res[i, j] = 1
    return res


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
        res.append(frozenset(ring_list))
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
        >>> mols = [Chem.MolFromSmiles('Cc1ccccc1'), Chem.MolFromSmiles('c1cccc2c1CCCC2')]
        >>> ccs = mols_to_cc(mols)

    """
    ccs = []
    for mol in mols:
        CC = CombinatorialComplex()

        # Atom
        for atom in mol.GetAtoms():
            CC.add_cell((atom.GetIdx(),), rank=0, symbol=atom.GetSymbol())

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
        return [np.array(), np.array(), np.array()]

    # Nodes
    nodes = CC.cells.hyperedge_dict[0]
    N = len(nodes)
    f = min(1, len(nodes[list(nodes.keys())[0]]))
    X = np.zeros((N, f))
    attributes_names = list(nodes[list(nodes.keys())[0]].keys())
    attributes_names.remove("weight")
    for k in list(nodes.keys()):
        node = list(k)[0]
        if not (attributes_names):
            X[node, 0] = 1
        else:
            for attr_id, attr in enumerate(attributes_names):
                X[node, attr_id] = nodes[k][attr]

    # Edges
    if 1 not in CC.cells.hyperedge_dict:
        return [X, np.array(), np.array()]
    edges = CC.cells.hyperedge_dict[1]
    A = np.zeros((N, N))
    for edge in list(edges.keys()):
        i, j = tuple(edge)
        A[i, j] = 1
        A[j, i] = 1

    # Rank-2 cells
    if 2 not in CC.cells.hyperedge_dict:
        return [X, A, np.array()]
    rank_2_cells = list(CC.cells.hyperedge_dict[2].keys())
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
            atom_symbol = atoms[atom]["symbol"]
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
        rank2 (torch.Tensor): batch of rank2 incidence matrices.
            B x (NC2) x K or B x C x (NC2) x K or (NC2) x K

    Returns:
        int: number of nodes
    """
    if len(rank2.shape) == 2:  # no batch
        nb_edges = rank2.shape[0]
    else:
        nb_edges = rank2.shape[1]
    N = int((1 + np.sqrt(1 + 8 * nb_edges)) / 2)
    return N


def get_rank2_flags(
    rank2: torch.Tensor, flags: torch.Tensor, N: int, d_min: int, d_max: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get flags for left and right nodes of rank2 cells.
    The left flag is 0 if the edge is not in the CC as a node is not.
    The right flag is 0 if the rank-2 cell is not in the CC as a node is not.

    Args:
        rank2 (torch.Tensor): batch of rank2 incidence matrices.
            B x (NC2) x K or B x C x (NC2) x K
        flags (torch.Tensor): 0-1 flags tensor. B x N
        N (int): number of nodes
        d_min (int): minimum dimension of rank2 cells
        d_max (int): maximum dimension of rank2 cells

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
    flags: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate noise for the rank-2 incidence matrix

    Args:
        x (torch.Tensor): input tensor
        flags (Optional[torch.Tensor], optional): optional flags. Defaults to None.

    Returns:
        torch.Tensor: generated noisy tensor
    """
    z = torch.randn_like(x)  # gaussian centered normal distribution
    z = mask_rank2(z, flags)
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


def ccs_to_tensors(
    cc_list: List[CombinatorialComplex], max_node_num: int, d_min: int, d_max: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a list of combinatorial complexes to two tensors, one for the adjacency matrices and one for the incidence matrices

    Args:
        cc_list (List[CombinatorialComplex]): list of combinatorial complexes
        max_node_num (int): max number of nodes in all the combinatorial complexes
        d_min (int): minimum dimension of rank2 cells
        d_max (int): maximum dimension of rank2 cells

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: adjacency matrices and rank2 incidence matrices
    """
    adjs_list = []
    rank2_list = []
    max_node_num = max_node_num  # memory issue

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
    cc: CombinatorialComplex, max_node_num: int, d_min: int, d_max: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a single combinatorial complex to a tuple of tensors, one for the adjacency matrix and one for the rank2 incidence matrix

    Args:
        cc (CombinatorialComplex): combinatorial complex to convert
        max_node_num (int): maximum number of nodes
        d_min (int): minimum dimension of rank2 cells
        d_max (int): maximum dimension of rank2 cells

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: adjacency matrix and rank2 incidence matrix
    """
    max_node_num = max_node_num  # memory issue

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

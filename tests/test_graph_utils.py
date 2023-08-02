#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_graph_utils.py: test functions for graph_utils.py
"""

from typing import List

import torch
import pytest
import networkx as nx
import numpy as np
from rdkit import Chem

from src.utils.errors import SymmetryError
from src.utils.graph_utils import (
    mask_x,
    mask_adjs,
    node_flags,
    init_features,
    gen_noise,
    quantize,
    quantize_mol,
    adjs_to_graphs,
    check_sym,
    pow_tensor,
    pad_adjs,
    graphs_to_tensor,
    graphs_to_adj,
    node_feature_to_matrix,
    nxs_to_mols,
)


def test_mask_x_with_flags() -> None:
    """Test the mask_x function with flags."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    flags = torch.tensor([[1.0, 0.0]])
    masked_x = mask_x(x, flags)
    expected_result = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])
    assert torch.allclose(masked_x, expected_result)


def test_mask_x_without_flags() -> None:
    """Test the mask_x function without flags."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    masked_x = mask_x(x)
    # Without flags, the result should be the same as input
    assert torch.allclose(masked_x, x)


def test_mask_adjs_with_flags() -> None:
    """Test the mask_adjs function with flags."""
    adjs = torch.tensor(
        [
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    flags = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    masked_adjs = mask_adjs(adjs, flags)
    expected_result = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    assert torch.allclose(masked_adjs, expected_result)
    # With channel
    adjs = torch.tensor(
        [
            [[[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]],
            [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        ]
    )
    flags = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    masked_adjs = mask_adjs(adjs, flags)
    expected_result = torch.tensor(
        [
            [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        ]
    )
    assert torch.allclose(masked_adjs, expected_result)


def test_mask_adjs_without_flags() -> None:
    """Test the mask_adjs function without flags."""
    adjs = torch.tensor(
        [
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    masked_adjs = mask_adjs(adjs)
    # Without flags, the result should be the same as input
    assert torch.allclose(masked_adjs, adjs)


def test_node_flags() -> None:
    """Test the node_flags function."""
    adj = torch.tensor([[0.0, 1e-7, 1e-7], [1e-7, 1.0, 1.0], [1e-7, 1.0, 1.0]])
    flags = node_flags(adj, eps=1e-5)
    expected_result = torch.tensor([0.0, 1.0, 1.0])
    assert torch.allclose(flags, expected_result)


def test_node_flags_different_threshold() -> None:
    """Test the node_flags function with a different threshold."""
    adj = torch.tensor([[0.0, 1e-7, 1e-7], [1e-7, 1.0, 1.0], [1e-7, 1.0, 1.0]])
    flags = node_flags(adj, eps=1e-8)
    expected_result = torch.tensor([1.0, 1.0, 1.0])
    assert torch.allclose(flags, expected_result)


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


def test_init_features_zeros() -> None:
    """Test the init_features function with zeros method."""
    init_method = "zeros"
    adjs = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        ]
    )
    # flag: tensor([[0., 1., 1., 1.]])
    nfeat = 10
    features = init_features(init_method, adjs, nfeat)
    assert torch.allclose(features, torch.zeros((1, 4, nfeat)))


def test_init_features_ones() -> None:
    """Test the init_features function with ones method."""
    init_method = "ones"
    adjs = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        ]
    )
    # flag: tensor([[0., 1., 1., 1.]])
    nfeat = 10
    features = init_features(init_method, adjs, nfeat)
    expected_features = torch.ones((1, 4, nfeat))
    expected_features[0][0] = torch.zeros(nfeat)
    assert torch.allclose(features, expected_features)


def test_init_features_deg() -> None:
    """Test the init_features function with deg method (one hot encoder on degree of nodes)."""
    init_method = "deg"
    adjs = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        ]
    )
    # flag: tensor([[0., 1., 1., 1.]])
    nfeat = 4
    features = init_features(init_method, adjs, nfeat)
    expected_result = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ]
    )
    assert torch.allclose(features, expected_result)


def test_init_features_not_implemented() -> None:
    """Test the init_features function with an invalid method."""
    init_method = "invalid_method"
    adjs = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        ]
    )
    nfeat = 10
    with pytest.raises(NotImplementedError):
        init_features(init_method, adjs, nfeat)


def test_gen_noise_sym() -> None:
    """Test the gen_noise function with sym=True."""
    adj = torch.rand((2, 3, 3))
    flags = torch.ones((2, 3), dtype=torch.float32)
    noise = gen_noise(adj, flags, sym=True)
    assert noise.shape == (2, 3, 3)
    assert torch.allclose(noise, noise.transpose(-1, -2))


def test_gen_noise_asym() -> None:
    """Test the gen_noise function with sym=False."""
    x = torch.rand((2, 3, 4))
    flags = torch.ones((2, 3), dtype=torch.float32)
    noise = gen_noise(x, flags, sym=False)
    assert noise.shape == (2, 3, 4)


def test_gen_noise_with_flags() -> None:
    """Test the gen_noise function with flags."""
    x = torch.rand((2, 3, 4))
    flags = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    noise = gen_noise(x, flags, sym=False)
    assert noise.shape == (2, 3, 4)
    assert torch.allclose(noise[0, 1], torch.zeros(4))
    assert torch.allclose(noise[1, 0], torch.zeros(4))
    assert torch.allclose(noise[1, 2], torch.zeros(4))


def test_quantize() -> None:
    """Test the quantize function."""
    t = torch.tensor([[0.1, 0.6], [0.4, 0.8]])
    thr = 0.5
    result = quantize(t, thr)
    expected_result = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    assert torch.allclose(result, expected_result)


def test_quantize_mol() -> None:
    """Test the quantize_mol function (quantize for molecules)."""
    adjs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.5, 1.0], [2.0, 3.0]]])
    quantized_adjs = quantize_mol(adjs)
    expected_result = np.array([[[1, 2], [3, 3]], [[2, 1], [2, 3]]])
    assert np.array_equal(quantized_adjs, expected_result)


def test_adjs_to_graphs() -> None:
    """Test the adjs_to_graphs function."""
    adjs = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ],
            [
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 0.0],
            ],
        ]
    )
    graph_list = adjs_to_graphs(adjs)
    assert len(graph_list) == 2
    assert isinstance(graph_list[0], nx.Graph)
    assert graph_list[0].number_of_nodes() == 3
    assert graph_list[1].number_of_nodes() == 4
    assert graph_list[0].number_of_edges() == 3
    assert graph_list[1].number_of_edges() == 6


def test_check_sym_pass() -> None:
    """Test the check_sym function (pass)."""
    adjs = torch.tensor([[[1.0, 2.0], [2.0, 3.0]], [[3.0, 4.0], [4.0, 5.0]]])
    check_sym(adjs)


def test_check_sym_fail() -> None:
    """Test the check_sym function (fail)."""
    adjs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    with pytest.raises(SymmetryError):
        check_sym(adjs)


def test_check_sym_fail_print_val() -> None:
    """Test the check_sym function (fail with print_val=True)."""
    adjs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    with pytest.raises(SymmetryError, match=r"Tensor not symmetric:"):
        check_sym(adjs, print_val=True)


def test_pow_tensor() -> None:
    """Test the pow_tensor function."""
    x = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)
    cnum = 3
    result = pow_tensor(x, cnum)
    expected_result = torch.tensor(
        [[[[1, 2], [3, 4]], [[7, 10], [15, 22]], [[37, 54], [81, 118]]]],
        dtype=torch.float32,
    )
    assert torch.allclose(result, expected_result)


def test_pad_adjs_pass() -> None:
    """Test the pad_adjs function (pass)."""
    ori_adj = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    node_number = 5
    result = pad_adjs(ori_adj, node_number)
    expected_result = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert np.array_equal(result, expected_result)


def test_pad_adjs_fail() -> None:
    """Test the pad_adjs function (fail)."""
    ori_adj = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    node_number = 2
    with pytest.raises(ValueError):
        pad_adjs(ori_adj, node_number)


def test_graphs_to_tensor() -> None:
    """Test the graphs_to_tensor function."""
    graph_list = [
        nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]).to_undirected(),
        nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2)]).to_undirected(),
    ]
    max_node_num = 5
    result = graphs_to_tensor(graph_list, max_node_num)
    expected_result = torch.tensor(
        [
            [
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(result, expected_result)


def test_graphs_to_adj() -> None:
    """Test the graphs_to_adj function."""
    graph = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]).to_undirected()
    max_node_num = 5
    result = graphs_to_adj(graph, max_node_num)
    expected_result = torch.tensor(
        [
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(result, expected_result)


def test_node_feature_to_matrix() -> None:
    """Test the node_feature_to_matrix function."""
    x = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)
    result = node_feature_to_matrix(x)
    expected_result = torch.tensor(
        [[[[1, 2, 1, 2], [1, 2, 3, 4]], [[3, 4, 1, 2], [3, 4, 3, 4]]]],
        dtype=torch.float32,
    )
    assert torch.allclose(result, expected_result)


def test_nxs_to_mols() -> None:
    """Test the nxs_to_mols function."""
    g = nx.Graph()
    # Add edge labels
    for atom_a, atom_b, bond_type in [
        (0, 1, 2),
        (1, 2, 1),
        (2, 3, 1),
        (2, 5, 1),
        (2, 8, 1),
        (3, 4, 1),
        (3, 8, 1),
        (4, 5, 1),
        (5, 6, 1),
        (6, 7, 1),
        (7, 8, 1),
    ]:
        g.add_edge(atom_a, atom_b, label=bond_type)
    # Add node labels
    for node, symbol in [
        (0, "O"),
        (1, "C"),
        (2, "C"),
        (3, "C"),
        (4, "C"),
        (5, "C"),
        (6, "C"),
        (7, "C"),
        (8, "N"),
    ]:
        g.nodes[node]["label"] = symbol
    # To undirected
    g = g.to_undirected()
    # Convert graph to mol
    mol = nxs_to_mols([g])[0]
    # Check if the molecule is correct by comparing the smiles
    smiles = Chem.MolToSmiles(mol)
    assert smiles == "O=CC12C3CCN1CC32"

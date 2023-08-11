#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_data_loader.py: test functions for data_loader.py
"""

import os
import pickle
from typing import List

import networkx as nx
import pytest
from easydict import EasyDict
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from torch.utils.data import DataLoader

from ccsd.src.utils.data_loader import (
    ccs_to_dataloader,
    dataloader,
    dataloader_cc,
    graphs_to_dataloader,
)


@pytest.fixture
def sample_graph_list() -> List[nx.Graph]:
    """Create a sample list of graphs for testing.

    Returns:
        List[nx.Graph]: sample list of graphs
    """
    G1 = nx.Graph()
    G1.add_edge(0, 1)
    G1.add_edge(1, 2)
    G1.add_edge(2, 3)
    G1.add_edge(3, 0)
    G1.add_edge(3, 4)
    G1.add_edge(4, 0)
    G2 = nx.Graph()
    G2.add_edge(0, 1)
    G2.add_edge(1, 2)
    G2.add_edge(2, 0)
    G2.add_edge(1, 3)
    G2.add_edge(2, 3)
    return [G1.to_undirected(), G2.to_undirected()]


def test_graphs_to_dataloader(sample_graph_list: List[nx.Graph]) -> None:
    """Test the graphs_to_dataloader function.

    Args:
        sample_graph_list (List[nx.Graph]): sample list of graphs
    """
    # Define a sample configuration for testing
    config = EasyDict(
        {
            "data": {
                "init": "ones",
                "max_node_num": 10,
                "max_feat_num": 5,
                "batch_size": 2,
            }
        }
    )

    # Call the function to be tested
    dataloader = graphs_to_dataloader(config, sample_graph_list)

    # Assert the output is a DataLoader object
    assert isinstance(dataloader, DataLoader)


@pytest.fixture
def sample_cc_list() -> List[CombinatorialComplex]:
    """Create a sample list of combinatorial complexes for testing.

    Returns:
        List[CombinatorialComplex]: sample list of combinatorial complexes
    """
    # Create a sample list of combinatorial complexes for testing
    CC1 = CombinatorialComplex()
    CC1.add_cell([0, 1], rank=1)
    CC1.add_cell([1, 2], rank=1)
    CC1.add_cell([2, 3], rank=1)
    CC1.add_cell([3, 0], rank=1)
    CC1.add_cell([3, 4], rank=1)
    CC1.add_cell([4, 0], rank=1)
    CC1.add_cell([3, 4, 0], rank=2)
    CC1.add_cell([0, 1, 2, 3], rank=2)
    CC2 = CombinatorialComplex()
    CC2.add_cell([0, 1], rank=1)
    CC2.add_cell([1, 2], rank=1)
    CC2.add_cell([2, 0], rank=1)
    CC2.add_cell([1, 3], rank=1)
    CC2.add_cell([3, 2], rank=1)
    CC2.add_cell([0, 1, 2], rank=2)
    CC2.add_cell([1, 2, 3], rank=2)
    return [CC1, CC2]


def test_ccs_to_dataloader(sample_cc_list: List[CombinatorialComplex]) -> None:
    """Test the ccs_to_dataloader function.

    Args:
        sample_cc_list (List[CombinatorialComplex]): sample list of combinatorial complexes
    """
    # Define a sample configuration for testing
    config = EasyDict(
        {
            "data": {
                "init": "ones",
                "max_node_num": 5,
                "d_min": 3,
                "d_max": 4,
                "max_feat_num": 5,
                "batch_size": 2,
            }
        }
    )

    # Call the function to be tested
    dataloader = ccs_to_dataloader(config, sample_cc_list)

    # Assert the output is a DataLoader object
    assert isinstance(dataloader, DataLoader)


def test_dataloader(sample_graph_list: List[nx.Graph]) -> None:
    """Test the dataloader function.

    Args:
        sample_graph_list (List[nx.Graph]): sample list of graphs
    """
    # Define a sample configuration for testing
    temp_dir = os.path.join(*["tests"])
    temp_data_name = "test_data"
    with open(os.path.join(temp_dir, f"{temp_data_name}.pkl"), "wb") as f:
        pickle.dump(sample_graph_list, f)
    config = EasyDict(
        {
            "data": {
                "init": "ones",
                "dir": temp_dir,
                "data": temp_data_name,
                "max_node_num": 5,
                "max_feat_num": 5,
                "batch_size": 1,
                "test_split": 0.5,
            }
        }
    )

    # Call the function to be tested
    train_dl, test_dl = dataloader(config)

    # Assert the output is a tuple of DataLoader objects
    assert isinstance(train_dl, DataLoader)
    assert isinstance(test_dl, DataLoader)

    # Check the function with get_cc_list=True
    train_graph_list, test_graph_list = dataloader(config, get_graph_list=True)
    assert isinstance(train_graph_list, list)
    assert isinstance(test_graph_list, list)
    assert all(isinstance(g, nx.Graph) for g in train_graph_list)
    assert all(isinstance(g, nx.Graph) for g in test_graph_list)

    # Clean up
    os.remove(os.path.join(temp_dir, f"{temp_data_name}.pkl"))


def test_dataloader_cc(sample_cc_list: List[CombinatorialComplex]) -> None:
    """Test the dataloader_cc function.

    Args:
        sample_cc_list (List[CombinatorialComplex]): sample list of combinatorial complexes
    """
    # Define a sample configuration for testing
    temp_dir = os.path.join(*["tests"])
    temp_data_name = "test_data"
    with open(os.path.join(temp_dir, f"{temp_data_name}.pkl"), "wb") as f:
        pickle.dump(sample_cc_list, f)
    config = EasyDict(
        {
            "data": {
                "init": "ones",
                "dir": temp_dir,
                "data": temp_data_name,
                "max_node_num": 5,
                "d_min": 3,
                "d_max": 4,
                "max_feat_num": 5,
                "batch_size": 1,
                "test_split": 0.5,
            }
        }
    )

    # Call the function to be tested
    train_dl, test_dl = dataloader_cc(config)

    # Assert the output is a tuple of DataLoader objects
    assert isinstance(train_dl, DataLoader)
    assert isinstance(test_dl, DataLoader)

    # Check the function with get_cc_list=True
    train_cc_list, test_cc_list = dataloader_cc(config, get_cc_list=True)
    assert isinstance(train_cc_list, list)
    assert isinstance(test_cc_list, list)
    assert all(isinstance(cc, CombinatorialComplex) for cc in train_cc_list)
    assert all(isinstance(cc, CombinatorialComplex) for cc in test_cc_list)

    # Clean up
    os.remove(os.path.join(temp_dir, f"{temp_data_name}.pkl"))

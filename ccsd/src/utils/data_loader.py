#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""data_loader.py: utility functions for loading the graph data (not molecular ones).

Only dataloader left untouched from Jo, J. & al (2022)
"""

import os
from typing import List, Tuple, Union

import networkx as nx
from easydict import EasyDict
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from torch.utils.data import DataLoader, TensorDataset

from ccsd.data.data_generators import load_dataset
from ccsd.src.utils.cc_utils import ccs_to_tensors
from ccsd.src.utils.graph_utils import graphs_to_tensor, init_features


def graphs_to_dataloader(config: EasyDict, graph_list: List[nx.Graph]) -> DataLoader:
    """Convert a list of graphs to a dataloader.

    Args:
        config (EasyDict): configuration to use
        graph_list (List[nx.Graph]): list of graphs to convert

    Returns:
        DataLoader: DataLoader object for the graphs
    """

    adjs_tensor = graphs_to_tensor(graph_list, config.data.max_node_num)
    x_tensor = init_features(config.data.init, adjs_tensor, config.data.max_feat_num)

    train_ds = TensorDataset(x_tensor, adjs_tensor)
    train_dl = DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True)
    return train_dl


def ccs_to_dataloader(
    config: EasyDict, cc_list: List[CombinatorialComplex]
) -> DataLoader:
    """Convert a list of combinatorial complexes to a dataloader.

    Args:
        config (EasyDict): configuration to use
        cc_list (List[CombinatorialComplex]): list of combinatorial complexes to convert

    Returns:
        DataLoader: DataLoader object for the combinatorial complexes
    """

    adjs_tensor, rank2_tensor = ccs_to_tensors(
        cc_list, config.data.max_node_num, config.data.d_min, config.data.d_max
    )
    x_tensor = init_features(config.data.init, adjs_tensor, config.data.max_feat_num)

    train_ds = TensorDataset(x_tensor, adjs_tensor, rank2_tensor)
    train_dl = DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True)
    return train_dl


def dataloader(
    config: EasyDict, get_graph_list: bool = False
) -> Union[Tuple[DataLoader, DataLoader], Tuple[List[nx.Graph], List[nx.Graph]]]:
    """Load the dataset and return the train and test dataloader for the given non-molecular dataset.

    Args:
        config (EasyDict): configuration to use
        get_graph_list (bool, optional): if True, the dataloader are lists of graphs. Defaults to False.

    Returns:
        Union[Tuple[DataLoader, DataLoader], Tuple[List[nx.Graph], List[nx.Graph]]]: train and test dataloader (tensors or lists of graphs)
    """
    graph_list = load_dataset(
        data_dir=os.path.join(config.folder, config.data.dir),
        file_name=config.data.data,
    )
    test_size = int(config.data.test_split * len(graph_list))
    train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]
    if get_graph_list:  # return dataloader as lists of graphs
        return train_graph_list, test_graph_list

    return graphs_to_dataloader(config, train_graph_list), graphs_to_dataloader(
        config, test_graph_list
    )


def dataloader_cc(
    config: EasyDict, get_cc_list: bool = False
) -> Union[
    Tuple[DataLoader, DataLoader],
    Tuple[List[CombinatorialComplex], List[CombinatorialComplex]],
]:
    """Load the dataset and return the train and test dataloader for the given non-molecular dataset.

    Args:
        config (EasyDict): configuration to use
        get_cc_list (bool, optional): if True, the dataloader are lists of combinatorial complexes. Defaults to False.

    Returns:
        Union[Tuple[DataLoader, DataLoader], Tuple[List[CombinatorialComplex], List[CombinatorialComplex]]]: train and test dataloader (tensors or lists of combinatorial complexes)
    """
    cc_list = load_dataset(
        data_dir=os.path.join(config.folder, config.data.dir),
        file_name=config.data.data,
    )
    test_size = int(config.data.test_split * len(cc_list))
    train_cc_list, test_cc_list = cc_list[test_size:], cc_list[:test_size]
    if get_cc_list:  # return dataloader as lists of combinatorial complexes
        return train_cc_list, test_cc_list

    return ccs_to_dataloader(config, train_cc_list), ccs_to_dataloader(
        config, test_cc_list
    )

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""data_loader.py: utility functions for loading the graph data (not molecular ones).
"""

from typing import List, Tuple, Union

import networkx as nx
from easydict import EasyDict
from torch.utils.data import TensorDataset, DataLoader

from data.data_generators import load_dataset
from src.utils.graph_utils import init_features, graphs_to_tensor


# TODO: MODIFY THIS FILE TO DATALOAD COMBINATORIAL COMPLEXES


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
    graph_list = load_dataset(data_dir=config.data.dir, file_name=config.data.data)
    test_size = int(config.data.test_split * len(graph_list))
    train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]
    if get_graph_list:  # return dataloader as lists of graphs
        return train_graph_list, test_graph_list

    return graphs_to_dataloader(config, train_graph_list), graphs_to_dataloader(
        config, test_graph_list
    )

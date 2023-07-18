#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""data_loader_mol.py: utility functions for loading the graph data (molecular ones).
"""

import os
from time import time
from typing import List, Tuple, Any, Callable, Union

import torch
import json
import numpy as np
import networkx as nx
from easydict import EasyDict
from torch.utils.data import DataLoader, Dataset


# TODO: MODIFY THIS FILE TO DATALOAD COMBINATORIAL COMPLEXES


def load_mol(filepath: str) -> List[Tuple[Any, Any]]:
    """Load molecular dataset from filepath.

    Adapted from GraphEBM

    Args:
        filepath (str): filepath to the dataset

    Raises:
        ValueError: raise an error if the filepath is invalid

    Returns:
        List[Tuple[Any, Any]]: list of tuples of (node features, adjacency matrix)
    """

    print(f"Loading file {filepath}")
    if not os.path.exists(filepath):
        raise ValueError(f"Invalid filepath {filepath} for dataset")
    load_data = np.load(filepath)
    result = []
    i = 0
    while True:
        key = f"arr_{i}"
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    return list(map(lambda x, a: (x, a), result[0], result[1]))


class MolDataset(Dataset):
    """Dataset object for molecular dataset."""

    def __init__(
        self,
        mols: List[Tuple[np.ndarray, np.ndarray]],
        transform: Callable[
            [Tuple[np.ndarray, np.ndarray]], Tuple[torch.Tensor, torch.Tensor]
        ],
    ) -> None:
        """Initialize the dataset.

        Args:
            mols (List[Tuple[np.ndarray, np.ndarray]]): list of tuples of (node features, adjacency matrix)
            transform (Callable[[Tuple[np.ndarray, np.ndarray]], Tuple[torch.Tensor, torch.Tensor]]): transform function that transforms the data into tensors with some preprocessing
        """
        self.mols = mols
        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: length of the dataset
        """
        return len(self.mols)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the item of the dataset at the given index.

        Args:
            idx (int): index of the item

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple of (node features, adjacency matrix) as tensors
        """
        return self.transform(self.mols[idx])


def get_transform_fn(
    dataset: str,
) -> Callable[[Tuple[np.ndarray, np.ndarray]], Tuple[torch.Tensor, torch.Tensor]]:
    """Get the transform function for the given dataset.

    Args:
        dataset (str): name of the dataset

    Raises:
        ValueError: raise an error if the dataset is invalid/unsupported

    Returns:
        Callable[[Tuple[np.ndarray, np.ndarray]], Tuple[torch.Tensor, torch.Tensor]]: transform function that transforms the data into tensors with some preprocessing
    """
    if dataset == "QM9":

        def transform(
            data: Tuple[np.ndarray, np.ndarray]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Transform data from QM9 (node matrix, adj matrix) into tensors with some preprocessing.

            Args:
                data (Tuple[np.ndarray, np.ndarray]): tuple of (node features, adjacency matrix)

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: tuple of (node features, adjacency matrix) as tensors
            """
            x, adj = data
            # the last place is for virtual nodes
            # 6: C, 7: N, 8: O, 9: F
            x_ = np.zeros((9, 5))
            indices = np.where(x >= 6, x - 6, 4)
            x_[np.arange(9), indices] = 1
            x = torch.tensor(x_).to(torch.float32)
            # single, double, triple and no-bond; the last channel is for virtual edges
            adj = np.concatenate(
                [adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0
            ).astype(np.float32)

            x = x[:, :-1]  # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
            adj = torch.tensor(
                adj.argmax(axis=0)
            )  # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
            # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            return x, adj

    else:
        raise ValueError(f"Invalid dataset {dataset}")

    return transform


def dataloader_mol(
    config: EasyDict, get_graph_list: bool = False
) -> Union[Tuple[DataLoader, DataLoader], Tuple[List[nx.Graph], List[nx.Graph]]]:
    """Load the dataset and return the train and test dataloader for the given molecular dataset.

    Args:
        config (EasyDict): configuration to use
        get_graph_list (bool, optional): if True, the dataloader are lists of graphs. Defaults to False.

    Returns:
        Union[Tuple[DataLoader, DataLoader], Tuple[List[nx.Graph], List[nx.Graph]]]: train and test dataloader (tensors or lists of graphs)
    """
    start_time = time()

    mols = load_mol(
        os.path.join(config.data.dir, f"{config.data.data.lower()}_kekulized.npz")
    )

    with open(
        os.path.join(config.data.dir, f"valid_idx_{config.data.data.lower()}.json")
    ) as f:
        test_idx = json.load(f)

    if config.data.data == "QM9":  # process QM9 differently
        test_idx = test_idx["valid_idxs"]
        test_idx = [int(i) for i in test_idx]

    train_idx = [i for i in range(len(mols)) if i not in test_idx]
    print(
        f"Number of training mols: {len(train_idx)} | Number of test mols: {len(test_idx)}"
    )

    train_mols = [mols[i] for i in train_idx]
    test_mols = [mols[i] for i in test_idx]

    # Create MolDataset objects
    train_dataset = MolDataset(train_mols, get_transform_fn(config.data.data))
    test_dataset = MolDataset(test_mols, get_transform_fn(config.data.data))

    if get_graph_list:
        train_mols_nx = [
            nx.from_numpy_matrix(np.array(adj)) for x, adj in train_dataset
        ]
        test_mols_nx = [nx.from_numpy_matrix(np.array(adj)) for x, adj in test_dataset]
        return train_mols_nx, test_mols_nx

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.data.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.data.batch_size, shuffle=True
    )

    print(f"{time() - start_time:.2f} sec elapsed for data loading")
    return train_dataloader, test_dataloader

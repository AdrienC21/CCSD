#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""data_loader_mol.py: utility functions for loading the graph data (molecular ones).

Only dataloader_mol left untouched from Jo, J. & al (2022)
"""

import json
import os
from time import perf_counter
from typing import Any, Callable, List, Tuple, Union

import networkx as nx
import numpy as np
import torch
from easydict import EasyDict
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ccsd.data.data_generators import load_dataset, save_dataset
from ccsd.src.utils.cc_utils import (
    cc_from_incidence,
    create_incidence_1_2,
    get_all_mol_rings,
    get_mol_from_x_adj,
)


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
    try:
        load_data = np.load(
            filepath, allow_pickle=True
        )  # allow pickle for complex data
    except:
        with open(filepath, "rb") as f:
            load_data = np.load(f, allow_pickle=True)
    if isinstance(
        load_data, np.ndarray
    ):  # if the data is a numpy array, convert it to dict
        load_data = load_data.item()
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
        transform: Union[
            Callable[
                [Tuple[np.ndarray, np.ndarray]], Tuple[torch.Tensor, torch.Tensor]
            ],
            Callable[
                [Tuple[np.ndarray, np.ndarray]],
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            ],
        ],
    ) -> None:
        """Initialize the dataset.

        Args:
            mols (List[Tuple[np.ndarray, np.ndarray]]): list of tuples of (node features, adjacency matrix)
            transform (Union[Callable[[Tuple[np.ndarray, np.ndarray]], Tuple[torch.Tensor, torch.Tensor]], Callable[[Tuple[np.ndarray, np.ndarray]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
                transform function that transforms the data into tensors with some preprocessing. Two tensors for
                graph-based modelisation and three tensors for combinatorial complex-based modelisation.
        """
        self.mols = mols
        self.transform = transform

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: length of the dataset
        """
        return len(self.mols)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Get the item of the dataset at the given index.

        Args:
            idx (int): index of the item

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: tuple of (node features, adjacency matrix) or (node features, adjacency matrix, rank2 incidence matrix) as tensors
        """
        return self.transform(self.mols[idx])

    def __repr__(self) -> str:
        """Return the string representation of the MolDataset class.

        Returns:
            str: the string representation of the MolDataset class
        """
        return self.__class__.__name__


def get_transform_fn(
    dataset: str,
    is_cc: bool = False,
    **kwargs: Any,
) -> Union[
    Callable[[Tuple[np.ndarray, np.ndarray]], Tuple[torch.Tensor, torch.Tensor]],
    Callable[
        [Tuple[np.ndarray, np.ndarray]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
]:
    """Get the transform function for the given dataset.

    Args:
        dataset (str): name of the dataset
        is_cc (bool, optional): if True, the transform function returns three tensors
            for combinatorial complexes modelisation. Defaults to False.

    Raises:
        ValueError: raise an error if the dataset is invalid/unsupported

    Returns:
        Union[Callable[[Tuple[np.ndarray, np.ndarray]], Tuple[torch.Tensor, torch.Tensor]], Callable[[Tuple[np.ndarray, np.ndarray]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
            transform function that transforms the data into tensors with some preprocessing. Two tensors
            for graph-based modelisation and three tensors for combinatorial complex-based modelisation.
    """
    if dataset == "QM9":
        if not (is_cc):

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

                x = x[
                    :, :-1
                ]  # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
                adj = torch.tensor(
                    adj.argmax(axis=0)
                )  # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
                # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
                adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
                return x, adj

        else:
            d_min = kwargs["d_min"]
            d_max = kwargs["d_max"]

            def transform(
                data: Tuple[np.ndarray, np.ndarray],
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                """Transform data from QM9 (node matrix, adj matrix) into tensors with some preprocessing.

                Args:
                    data (Tuple[np.ndarray, np.ndarray]): tuple of (node features, adjacency matrix)

                Returns:
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple of (node features, adjacency matrix, rank2 incidence matrix) as tensors
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

                x = x[
                    :, :-1
                ]  # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
                adj = torch.tensor(
                    adj.argmax(axis=0)
                )  # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
                # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
                adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)

                # rank2 incidence matrix
                mol = get_mol_from_x_adj(x, adj)
                rings = get_all_mol_rings(mol)
                rings = {ring: {} for ring in rings}  # convert to dict
                rank2 = create_incidence_1_2(
                    x.shape[0], adj, d_min, d_max, two_rank_cells=rings
                )
                rank2 = torch.tensor(rank2).to(torch.float32)
                return x, adj, rank2

    elif dataset == "ZINC250k":
        if not (is_cc):

            def transform(
                data: Tuple[np.ndarray, np.ndarray],
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                """Transform data from ZINC250k (node matrix, adj matrix) into tensors with some preprocessing.

                Args:
                    data (Tuple[np.ndarray, np.ndarray]): tuple of (node features, adjacency matrix)

                Returns:
                    Tuple[torch.Tensor, torch.Tensor]: tuple of (node features, adjacency matrix) as tensors
                """
                x, adj = data
                # the last place is for virtual nodes
                # 6: C, 7: N, 8: O, 9: F, 15: P, 16: S, 17: Cl, 35: Br, 53: I
                zinc250k_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
                x_ = np.zeros((38, 10), dtype=np.float32)
                for i in range(38):
                    ind = zinc250k_atomic_num_list.index(x[i])
                    x_[i, ind] = 1.0
                x = torch.tensor(x_).to(torch.float32)
                # single, double, triple and no-bond; the last channel is for virtual edges
                adj = np.concatenate(
                    [adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0
                ).astype(np.float32)

                x = x[
                    :, :-1
                ]  # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
                adj = torch.tensor(
                    adj.argmax(axis=0)
                )  # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
                # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
                adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
                return x, adj

        else:
            d_min = kwargs["d_min"]
            d_max = kwargs["d_max"]

            def transform(
                data: Tuple[np.ndarray, np.ndarray],
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                """Transform data from ZINC250k (node matrix, adj matrix) into tensors with some preprocessing.

                Args:
                    data (Tuple[np.ndarray, np.ndarray]): tuple of (node features, adjacency matrix)

                Returns:
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple of (node features, adjacency matrix, rank2 incidence matrix) as tensors
                """
                x, adj = data
                # the last place is for virtual nodes
                # 6: C, 7: N, 8: O, 9: F, 15: P, 16: S, 17: Cl, 35: Br, 53: I
                zinc250k_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
                x_ = np.zeros((38, 10), dtype=np.float32)
                for i in range(38):
                    ind = zinc250k_atomic_num_list.index(x[i])
                    x_[i, ind] = 1.0
                x = torch.tensor(x_).to(torch.float32)
                # single, double, triple and no-bond; the last channel is for virtual edges
                adj = np.concatenate(
                    [adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0
                ).astype(np.float32)

                x = x[
                    :, :-1
                ]  # 9, 5 (the last place is for vitual nodes) -> 9, 4 (38, 9)
                adj = torch.tensor(
                    adj.argmax(axis=0)
                )  # 4, 9, 9 (the last place is for vitual edges) -> 9, 9 (38, 38)
                # 0, 1, 2, 3 -> 1, 2, 3, 0; now virtual edges are denoted as 0
                adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)

                # rank2 incidence matrix
                mol = get_mol_from_x_adj(x, adj)
                rings = get_all_mol_rings(mol)
                rings = {ring: {} for ring in rings}  # convert to dict
                rank2 = create_incidence_1_2(
                    x.shape[0], adj, d_min, d_max, two_rank_cells=rings
                )
                rank2 = torch.tensor(rank2).to(torch.float32)
                return x, adj, rank2

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
    dataset_name = f"{config.data.data}_graphs_{get_graph_list}"
    data_dir = os.path.join(config.folder, config.data.dir)
    if os.path.exists(os.path.join(data_dir, f"{dataset_name}_train.pkl")):
        # Load the data
        print("Loading existing files...")
        train = load_dataset(data_dir=data_dir, file_name=f"{dataset_name}_train")
        test = load_dataset(data_dir=data_dir, file_name=f"{dataset_name}_test")
        return train, test
    # If the data does not exist, create it
    start_time = perf_counter()

    mols = load_mol(
        os.path.join(
            config.folder, config.data.dir, f"{config.data.data.lower()}_kekulized.npz"
        )
    )

    with open(
        os.path.join(
            config.folder, config.data.dir, f"valid_idx_{config.data.data.lower()}.json"
        )
    ) as f:
        test_idx = json.load(f)

    if config.data.data == "QM9":  # process QM9 differently
        test_idx = test_idx["valid_idxs"]
        test_idx = [int(i) for i in test_idx]

    test_idx = set(test_idx)  # convert to set to speed up the process

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
        print("Loading train graphs...")
        train_mols_nx = []
        for i in tqdm(range(len(train_dataset))):
            _, adj = train_dataset[i]
            train_mols_nx.append(nx.from_numpy_matrix(np.array(adj)))
        print("Loading test graphs...")
        test_mols_nx = []
        for i in tqdm(range(len(test_dataset))):
            _, adj = test_dataset[i]
            test_mols_nx.append(nx.from_numpy_matrix(np.array(adj)))
        save_dataset(
            data_dir=data_dir,
            obj=train_mols_nx,
            save_name=f"{dataset_name}_train",
            save_txt=False,
        )
        save_dataset(
            data_dir=data_dir,
            obj=test_mols_nx,
            save_name=f"{dataset_name}_test",
            save_txt=False,
        )
        print(f"{perf_counter() - start_time:.2f} sec elapsed for data loading")
        return train_mols_nx, test_mols_nx

    print("Loading train dataloader...")
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.data.batch_size, shuffle=True
    )
    print("Loading test dataloader...")
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.data.batch_size, shuffle=True
    )
    print(f"{perf_counter() - start_time:.2f} sec elapsed for data loading")
    return train_dataloader, test_dataloader


def dataloader_mol_cc(
    config: EasyDict, get_cc_list: bool = False
) -> Union[
    Tuple[DataLoader, DataLoader],
    Tuple[List[CombinatorialComplex], List[CombinatorialComplex]],
]:
    """Load the dataset and return the train and test dataloader for the given molecular dataset.

    Args:
        config (EasyDict): configuration to use
        get_cc_list (bool, optional): if True, the dataloader are lists of combinatorial complexes. Defaults to False.

    Returns:
        Union[Tuple[DataLoader, DataLoader], Tuple[List[CombinatorialComplex], List[CombinatorialComplex]]]: train and test dataloader (tensors or lists of combinatorial complexes)
    """
    dataset_name = f"{config.data.data}_cc_{get_cc_list}"
    data_dir = os.path.join(config.folder, config.data.dir)
    if os.path.exists(os.path.join(data_dir, f"{dataset_name}_train.pkl")):
        # Load the data
        print("Loading existing files...")
        train = load_dataset(data_dir=data_dir, file_name=f"{dataset_name}_train")
        test = load_dataset(data_dir=data_dir, file_name=f"{dataset_name}_test")
        return train, test
    # If the data does not exist, create it
    start_time = perf_counter()

    mols = load_mol(
        os.path.join(
            config.folder, config.data.dir, f"{config.data.data.lower()}_kekulized.npz"
        )
    )

    with open(
        os.path.join(
            config.folder, config.data.dir, f"valid_idx_{config.data.data.lower()}.json"
        )
    ) as f:
        test_idx = json.load(f)

    if config.data.data == "QM9":  # process QM9 differently
        test_idx = test_idx["valid_idxs"]
        test_idx = [int(i) for i in test_idx]

    test_idx = set(test_idx)  # convert to set to speed up the process

    train_idx = [i for i in range(len(mols)) if i not in test_idx]
    print(
        f"Number of training mols: {len(train_idx)} | Number of test mols: {len(test_idx)}"
    )

    train_mols = [mols[i] for i in train_idx]
    test_mols = [mols[i] for i in test_idx]

    # Create MolDataset objects
    train_dataset = MolDataset(
        train_mols,
        get_transform_fn(
            config.data.data,
            is_cc=True,
            d_min=config.data.d_min,
            d_max=config.data.d_max,
        ),
    )
    test_dataset = MolDataset(
        test_mols,
        get_transform_fn(
            config.data.data,
            is_cc=True,
            d_min=config.data.d_min,
            d_max=config.data.d_max,
        ),
    )

    if get_cc_list:
        print("Loading train combinatorial complexes...")
        train_mols_cc = []
        for i in tqdm(range(len(train_dataset))):
            x, adj, rank2 = train_dataset[i]
            train_mols_cc.append(
                cc_from_incidence(
                    [x, adj, rank2],
                    config.data.d_min,
                    config.data.d_max,
                    is_molecule=True,
                )
            )
        print("Loading test combinatorial complexes...")
        test_mols_cc = []
        for i in tqdm(range(len(test_dataset))):
            x, adj, rank2 = test_dataset[i]
            test_mols_cc.append(
                cc_from_incidence(
                    [x, adj, rank2],
                    config.data.d_min,
                    config.data.d_max,
                    is_molecule=True,
                )
            )
        save_dataset(
            data_dir=data_dir,
            obj=train_mols_cc,
            save_name=f"{dataset_name}_train",
            save_txt=False,
        )
        save_dataset(
            data_dir=data_dir,
            obj=test_mols_cc,
            save_name=f"{dataset_name}_test",
            save_txt=False,
        )
        print(f"{perf_counter() - start_time:.2f} sec elapsed for data loading")
        return train_mols_cc, test_mols_cc

    print("Loading train dataloader...")
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.data.batch_size, shuffle=True
    )
    print("Loading test dataloader...")
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.data.batch_size, shuffle=True
    )
    print(f"{perf_counter() - start_time:.2f} sec elapsed for data loading")
    return train_dataloader, test_dataloader

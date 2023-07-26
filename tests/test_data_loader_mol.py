#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_data_loader_mol.py: test functions for data_loader_mol.py
"""

import os
from typing import List, Tuple

import pytest
import pathlib
import numpy as np
import torch

from src.utils.data_loader_mol import load_mol, MolDataset, get_transform_fn


@pytest.fixture
def sample_data() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Define a fixture with sample data for testing

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: sample data
    """
    x = np.array([[6, 7, 8], [6, 9, 7]])
    adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return [(x, adj)]


def test_load_mol_valid_filepath(
    sample_data: List[Tuple[np.ndarray, np.ndarray]], tmpdir: pathlib.Path
):
    # Create a temporary file and save the sample data in it
    filepath = tmpdir.join("sample_data.npy")

    # Save the sample data in the temporary file
    to_save = {f"arr_{i}": [sample[i] for sample in sample_data] for i in range(2)}
    # arr_0: all the node features, arr_1: all the adjacency matrices
    with open(filepath, "wb") as f:
        np.save(f, to_save, allow_pickle=True)

    # Call the function to be tested
    result = load_mol(filepath)

    # Assert the result matches the sample data
    # For the node features
    assert all((r[0] == s[0]).all() for r, s in zip(result, sample_data))
    # For the adjacency matrix
    assert all((r[1] == s[1]).all() for r, s in zip(result, sample_data))

    # Clean up the temporary file
    os.remove(filepath)


def test_load_mol_invalid_filepath() -> None:
    """Test the load_mol function with an invalid filepath"""
    with pytest.raises(ValueError):
        load_mol("invalid_filepath.npy")


def test_mol_dataset() -> None:
    """Test the MolDataset class"""

    # Create a sample transform function for testing
    def transform(
        data: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, adj = data
        x = torch.tensor(x).to(torch.float32)
        adj = torch.tensor(adj).to(torch.float32)
        return x, adj

    # Create a sample MolDataset object with one sample data point
    sample_data = [(np.array([[6, 7], [6, 9]]), np.array([[0, 1], [1, 0]]))]
    dataset = MolDataset(sample_data, transform)

    # Test __len__
    assert len(dataset) == 1

    # Test __getitem__
    data_item = dataset[0]
    assert isinstance(data_item[0], torch.Tensor)
    assert isinstance(data_item[1], torch.Tensor)


def test_get_transform_fn() -> None:
    """Test the get_transform_fn function"""
    # Test for valid dataset (QM9 here)
    transform_fn = get_transform_fn("QM9")
    assert callable(transform_fn)

    # Test for valid dataset with combinatorial complexes
    kwargs = {"d_min": 3, "d_max": 4}
    transform_fn_cc = get_transform_fn("QM9", is_cc=True, **kwargs)
    assert callable(transform_fn_cc)

    # Test for invalid dataset
    with pytest.raises(ValueError):
        get_transform_fn("InvalidDataset")

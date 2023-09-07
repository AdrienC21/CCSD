#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""run_plot_original_datasets.py: Code to plot samples from the original datasets.
"""

import argparse
import math
import os
import pickle
import random
import sys
import warnings

sys.path.insert(0, os.getcwd())

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Draw

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
# Parameters to make graph plots look nicer.
options = {"node_size": 2, "edge_color": "black", "linewidths": 1, "width": 0.5}


def plot_qm9(args: argparse.ArgumentParser) -> None:
    """Plot samples from the QM9 dataset.

    Args:
        args (argparse.ArgumentParser): arguments
    """
    folder = args.folder
    N = args.N
    shuffle_mol = args.shuffle_mol

    mols_df = pd.read_csv(os.path.join(folder, "data", "qm9.csv"))
    smiles = mols_df["SMILES1"].tolist()
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    if shuffle_mol:
        random.shuffle(mols)

    max_num = 16
    img_c = int(math.ceil(np.sqrt(max_num)))
    figure = plt.figure()

    for i in range(max_num):
        idx = i + max_num * N
        mol = mols[idx]

        assert isinstance(
            mol, Chem.Mol
        ), "elements should be molecules"  # check if we have a molecule

        ax = plt.subplot(img_c, img_c, i + 1)
        mol_img = Draw.MolToImage(mol, size=(300, 300))
        ax.imshow(mol_img)
        title_str = f"{Chem.MolToSmiles(mol)}"
        ax.title.set_text(title_str)
        ax.set_axis_off()
    figure.suptitle("Original QM9 dataset")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    fig_dir = os.path.join(folder, "analysis")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(
        os.path.join(fig_dir, f"original_qm9.png"),
        bbox_inches="tight",
        dpi=300,
        transparent=False,
    )
    plt.close()


def plot_graph_data(args: argparse.ArgumentParser, name: str) -> None:
    """Plot samples from the graph datasets.

    Args:
        args (argparse.ArgumentParser): arguments
        name (str): name of the dataset
    """
    with open(os.path.join(args.folder, "data", f"{name}.pkl"), "rb") as f:
        graphs = pickle.load(f)
    max_num = 16
    img_c = int(math.ceil(np.sqrt(max_num)))
    figure = plt.figure()

    folder = args.folder
    N = args.N

    for i in range(max_num):
        idx = i + max_num * N
        if not isinstance(graphs[idx], nx.Graph):
            G = graphs[idx].g.copy()
        else:
            G = graphs[idx].copy()
        assert isinstance(G, nx.Graph)  # check if we have a networkx graph
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)

        ax = plt.subplot(img_c, img_c, i + 1)
        title_str = f"e={e - l}, n={v}"
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, **options)
        ax.title.set_text(title_str)
        ax.set_axis_off()
    figure.suptitle(f"Original {name.replace('_', ' ')} dataset")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    fig_dir = os.path.join(folder, "analysis")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(
        os.path.join(fig_dir, f"original_{name}.png"),
        bbox_inches="tight",
        dpi=300,
        transparent=False,
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the original datasets")
    parser.add_argument(
        "--folder",
        type=str,
        default="./",
        help="Directory of the root of the CCSD project",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=0,
        help="Shift the index of the graphs to plot by N",
    )
    parser.add_argument(
        "--shuffle_mol",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, shuffle the list of molecules",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_known_args()[0]

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # QM9
    plot_qm9(args)
    # Grid small
    plot_graph_data(args, "grid_small")
    # Community small
    plot_graph_data(args, "community_small")
    # Ego small
    plot_graph_data(args, "ego_small")
    # Enzymes small
    plot_graph_data(args, "ENZYMES_small")

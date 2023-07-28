#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plot.py: utility functions for plotting.
"""

import math
import os
import warnings
from typing import List, Optional, Union, Dict, Any

import matplotlib
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import hypernetx as hnx  # to visalize CC of dim 2
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from rdkit import Chem
from rdkit.Chem import Draw


warnings.filterwarnings(
    "ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning
)

# Parameters to make graph plots look nicer.
options = {"node_size": 2, "edge_color": "black", "linewidths": 1, "width": 0.5}


def save_fig(
    save_dir: Optional[str] = None,
    title: str = "fig",
    dpi: int = 300,
    is_sample: bool = True,
) -> None:
    """Function to adjust the figure and save it.

    Args:
        save_dir (Optional[str], optional): directory to save the figures. Defaults to None.
        title (str, optional): name of the file. Defaults to "fig".
        dpi (int, optional): DPI (Dots per Inch). Defaults to 300.
        is_sample (bool, optional): whether the figure is generated during the sample phase. Defaults to True.
    """
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save_dir is None:
        plt.show()
    else:
        if is_sample:
            fig_dir = os.path.join(*["samples", "fig", save_dir])
        else:
            fig_dir = os.path.join(*[save_dir, "fig"])
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(
            os.path.join(fig_dir, title),
            bbox_inches="tight",
            dpi=dpi,
            transparent=False,
        )
        plt.close()
    return


def plot_graphs_list(
    graphs: List[Union[nx.Graph, Dict[str, Any]]],
    title: str = "title",
    max_num: int = 16,
    save_dir: Optional[str] = None,
    N: int = 0,
) -> None:
    """_summary_

    Args:
        graphs (List[Union[nx.Graph, Dict[str, Any]]]): graphs to plot
        title (str, optional): title of the plot. Defaults to "title".
        max_num (int, optional): number of graphs to plot (must lower or equal than batch size). Defaults to 16.
        save_dir (Optional[str], optional): directory to save the figures. Defaults to None.
        N (int, optional): parameter to skip the first graphs of the list. Defaults to 0.
    """
    batch_size = len(graphs)
    max_num = min(batch_size, max_num)
    img_c = int(math.ceil(np.sqrt(max_num)))
    figure = plt.figure()

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
    figure.suptitle(title)

    save_fig(save_dir=save_dir, title=title)


def save_graph_list(
    log_folder_name: str, exp_name: str, gen_graph_list: List[nx.Graph]
) -> str:
    """Save the generated graphs in a pickle file.

    Args:
        log_folder_name (str): name of the folder where the pickle file will be saved
        exp_name (str): name of the experiment
        gen_graph_list (List[nx.Graph]): list of generated graphs

    Returns:
        str: path to the pickle file
    """
    if not (os.path.isdir("./samples/pkl/{}".format(log_folder_name))):
        os.makedirs(os.path.join("./samples/pkl/{}".format(log_folder_name)))
    with open("./samples/pkl/{}/{}.pkl".format(log_folder_name, exp_name), "wb") as f:
        pickle.dump(obj=gen_graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    save_dir = "./samples/pkl/{}/{}.pkl".format(log_folder_name, exp_name)
    return save_dir


def plot_cc_list(
    ccs: List[Union[CombinatorialComplex, Dict[str, Any]]],
    title: str = "title",
    max_num: int = 16,
    save_dir: Optional[str] = None,
    N: int = 0,
) -> None:
    """Plot a list of combinatorial complexes (represented here as hypergraphs), using hypernetx,
    for complexes of dimension 2.

    Args:
        ccs (List[Union[CombinatorialComplexes, Dict[str, Any]]]): combinatorial complexes to plot
        title (str, optional): title of the plot. Defaults to "title".
        max_num (int, optional): number of combinatorial complexes to plot (must lower or equal than batch size). Defaults to 16.
        save_dir (Optional[str], optional): directory to save the figures. Defaults to None.
        N (int, optional): parameter to skip the first graphs of the list. Defaults to 0.
    """
    batch_size = len(ccs)
    max_num = min(batch_size, max_num)
    img_c = int(math.ceil(np.sqrt(max_num)))
    figure = plt.figure()

    for i in range(max_num):
        idx = i + max_num * N

        if isinstance(ccs[idx], dict):
            cc = ccs[idx].get("cc", None)
        else:
            cc = ccs[idx]

        assert isinstance(
            cc, CombinatorialComplex
        ), "elements should be combinatorial complexes"  # check if we have a combinatorial complex

        v = len(cc.skeleton(0))  # number of vertices (rank 0)
        e = len(cc.skeleton(1))  # number of edges (rank 1)
        f = len(cc.skeleton(2))  # number of faces (rank 2)
        # Isolated nodes removed from the plot automatically as we use the edges/faces
        # Same for self loops as they can't be represented in a CC
        edges = cc.skeleton(1)
        scenes = {i: (tuple([str(n) for n in edge])) for i, edge in enumerate(edges)}
        scenes.update(
            {
                i + e: (tuple([str(n) for n in face]))
                for i, face in enumerate(cc.skeleton(2))
            }
        )
        H = hnx.Hypergraph(scenes)

        ax = plt.subplot(img_c, img_c, i + 1)
        hnx.drawing.draw(H, with_edge_labels=False, with_node_labels=False, ax=ax)
        title_str = f"n={v}, e={e}, f={f}"
        ax.title.set_text(title_str)
    figure.suptitle(title)

    save_fig(save_dir=save_dir, title=title)


def save_cc_list(
    log_folder_name: str, exp_name: str, gen_cc_list: List[CombinatorialComplex]
) -> str:
    """Save the generated combinatorial complexes in a pickle file.

    Args:
        log_folder_name (str): name of the folder where the pickle file will be saved
        exp_name (str): name of the experiment
        gen_cc_list (List[CombinatorialComplex]): list of generated ccs

    Returns:
        str: path to the pickle file
    """
    if not (os.path.isdir("./samples/pkl/{}".format(log_folder_name))):
        os.makedirs(os.path.join("./samples/pkl/{}".format(log_folder_name)))
    with open("./samples/pkl/{}/{}.pkl".format(log_folder_name, exp_name), "wb") as f:
        pickle.dump(obj=gen_cc_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    save_dir = "./samples/pkl/{}/{}.pkl".format(log_folder_name, exp_name)
    return save_dir


def plot_molecule_list(
    mols: List[Chem.Mol],
    title: str = "title",
    max_num: int = 16,
    save_dir: Optional[str] = None,
    N: int = 0,
) -> None:
    """Plot a list of molecules, using rdkit.

    Args:
        mols (List[Chem.Mol]): molecules to plot
        title (str, optional): title of the plot. Defaults to "title".
        max_num (int, optional): number of molecules to plot (must lower or equal than batch size). Defaults to 16.
        save_dir (Optional[str], optional): directory to save the figures. Defaults to None.
        N (int, optional): parameter to skip the first graphs of the list. Defaults to 0.
    """
    batch_size = len(mols)
    max_num = min(batch_size, max_num)
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
    figure.suptitle(title)

    save_fig(save_dir=save_dir, title=title)


def save_molecule_list(
    log_folder_name: str, exp_name: str, gen_mol_list: List[Chem.Mol]
) -> str:
    """Save the generated molecules in a pickle file.

    Args:
        log_folder_name (str): name of the folder where the pickle file will be saved
        exp_name (str): name of the experiment
        gen_mol_list (List[Chem.Mol]): list of generated molecules

    Returns:
        str: path to the pickle file
    """
    if not (os.path.isdir("./samples/pkl/{}".format(log_folder_name))):
        os.makedirs(os.path.join("./samples/pkl/{}".format(log_folder_name)))
    with open("./samples/pkl/{}/{}.pkl".format(log_folder_name, exp_name), "wb") as f:
        pickle.dump(obj=gen_mol_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    save_dir = "./samples/pkl/{}/{}.pkl".format(log_folder_name, exp_name)
    return save_dir


def plot_lc(
    learning_curves: Dict[str, List[float]],
    f_dir: str = "./",
    filename: str = "learning_curves",
    cols: int = 3,
) -> None:
    """Plot the learning curves.

    Args:
        learning_curves (Dict[str, List[float]]): dictionary containing the learning curves
        f_dir (str, optional): directory to save the figure. Defaults to "./".
        filename (str, optional): name of the figure. Defaults to "learning_curves".
        cols (int, optional): number of columns in the figure. Defaults to 3.
    """
    rows = int(math.ceil(len(learning_curves) / cols))
    figure = plt.figure(figsize=(20, 10))
    for i, (curve_name, curve) in enumerate(learning_curves.items()):
        curve_name = curve_name.replace("_", " ")  # make the title more readable
        ax = plt.subplot(rows, cols, i + 1)
        ax.plot(curve)
        ax.title.set_text(curve_name)
    figure.suptitle("Learning curves")

    save_fig(save_dir=f_dir, title=filename, is_sample=False)

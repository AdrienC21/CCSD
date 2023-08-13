#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""plot.py: utility functions for plotting.
"""

import math
import os
import pickle
import warnings
from typing import Any, Dict, FrozenSet, List, Optional, Union

import hypernetx as hnx  # to visalize CC of dim 2
import imageio.v3 as imageio
import kaleido  # import kaleido FIRST to avoid any conflicts
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly
import plotly.graph_objs as go
import torch
from easydict import EasyDict
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from toponetx.classes.combinatorial_complex import CombinatorialComplex
from tqdm import tqdm

from .graph_utils import adjs_to_graphs, quantize
from .mol_utils import construct_mol

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

# Parameters to make graph plots look nicer.
options = {"node_size": 2, "edge_color": "black", "linewidths": 1, "width": 0.5}


def save_fig(
    config: EasyDict,
    save_dir: Optional[str] = None,
    title: str = "fig",
    dpi: int = 300,
    is_sample: bool = True,
) -> None:
    """Function to adjust the figure and save it.

    Adapted from Jo, J. & al (2022)

    Args:
        config (EasyDict): configuration file
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
            fig_dir = os.path.join(*[config.folder, "samples", "fig", save_dir])
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
    config: EasyDict,
    graphs: List[Union[nx.Graph, Dict[str, Any]]],
    title: str = "title",
    max_num: int = 16,
    save_dir: Optional[str] = None,
    N: int = 0,
) -> None:
    """Plot a list of graphs.

    Adapted from Jo, J. & al (2022)

    Args:
        config (EasyDict): configuration file
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
        ax.set_axis_off()
    figure.suptitle(title)

    save_fig(config=config, save_dir=save_dir, title=title, is_sample=True)


def save_graph_list(
    config: EasyDict,
    log_folder_name: str,
    exp_name: str,
    gen_graph_list: List[nx.Graph],
) -> str:
    """Save the generated graphs in a pickle file.

    Adapted from Jo, J. & al (2022)

    Args:
        config (EasyDict): configuration file
        log_folder_name (str): name of the folder where the pickle file will be saved
        exp_name (str): name of the experiment
        gen_graph_list (List[nx.Graph]): list of generated graphs

    Returns:
        str: path to the pickle file
    """
    path = os.path.join(*[config.folder, "samples", "pkl", log_folder_name])
    if not (os.path.isdir(path)):
        os.makedirs(path)
    save_dir = os.path.join(*[path, exp_name])
    with open(save_dir, "wb") as f:
        pickle.dump(obj=gen_graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    return save_dir


def plot_cc_list(
    config: EasyDict,
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
        ax.set_axis_off()
    figure.suptitle(title)

    save_fig(config=config, save_dir=save_dir, title=title, is_sample=True)


def save_cc_list(
    config: EasyDict,
    log_folder_name: str,
    exp_name: str,
    gen_cc_list: List[CombinatorialComplex],
) -> str:
    """Save the generated combinatorial complexes in a pickle file.

    Args:
        config (EasyDict): configuration file
        log_folder_name (str): name of the folder where the pickle file will be saved
        exp_name (str): name of the experiment
        gen_cc_list (List[CombinatorialComplex]): list of generated ccs

    Returns:
        str: path to the pickle file
    """
    path = os.path.join(*[config.folder, "samples", "pkl", log_folder_name])
    if not (os.path.isdir(path)):
        os.makedirs(path)
    save_dir = os.path.join(*[path, exp_name])
    with open(save_dir, "wb") as f:
        pickle.dump(obj=gen_cc_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    return save_dir


def plot_molecule_list(
    config: EasyDict,
    mols: List[Chem.Mol],
    title: str = "title",
    max_num: int = 16,
    save_dir: Optional[str] = None,
    N: int = 0,
) -> None:
    """Plot a list of molecules, using rdkit.

    Args:
        config (EasyDict): configuration file
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
        ax.set_axis_off()
    figure.suptitle(title)

    save_fig(config=config, save_dir=save_dir, title=title, is_sample=True)


def save_molecule_list(
    config: EasyDict, log_folder_name: str, exp_name: str, gen_mol_list: List[Chem.Mol]
) -> str:
    """Save the generated molecules in a pickle file.

    Args:
        config (EasyDict): configuration file
        log_folder_name (str): name of the folder where the pickle file will be saved
        exp_name (str): name of the experiment
        gen_mol_list (List[Chem.Mol]): list of generated molecules

    Returns:
        str: path to the pickle file
    """
    path = os.path.join(*[config.folder, "samples", "pkl", log_folder_name])
    if not (os.path.isdir(path)):
        os.makedirs(path)
    save_dir = os.path.join(*[path, exp_name])
    with open(save_dir, "wb") as f:
        pickle.dump(obj=gen_mol_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    return save_dir


def plot_lc(
    config: EasyDict,
    learning_curves: Dict[str, List[float]],
    f_dir: str = "./",
    filename: str = "learning_curves",
    cols: int = 3,
) -> None:
    """Plot the learning curves.

    Args:
        config (EasyDict): configuration file
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

    save_fig(config=config, save_dir=f_dir, title=filename, is_sample=False)


def plot_3D_molecule(
    molecule: Chem.Mol,
    atomic_radii: Optional[Dict[str, float]] = None,
    cpk_colors: Optional[Dict[str, str]] = None,
) -> plotly.graph_objs.Figure:
    """Creates a 3D plot of the molecule.

    Args:
        molecule (Chem.Mol): The RDKit molecule to plot.
        atomic_radii (Optional[Dict[str, float]], optional): Dictionary mapping atomic symbols to atomic radii. Defaults to None.
        cpk_colors (Optional[Dict[str, str]], optional): Dictionary mapping atomic symbols to CPK colors. Defaults to None.

    Returns:
        plotly.graph_objs.Figure: The 3D plotly figure of the molecule.
    """
    # Default atomic radii from https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
    if atomic_radii is None:
        atomic_radii = {"C": 0.77, "F": 0.71, "H": 0.38, "N": 0.75, "O": 0.73}
    # Default CPK colors from https://en.wikipedia.org/wiki/CPK_coloring
    if cpk_colors is None:
        cpk_colors = {"C": "black", "F": "green", "H": "white", "N": "blue", "O": "red"}

    # Generate 3D coordinates if not already present
    if not molecule.GetNumConformers():
        AllChem.EmbedMolecule(molecule, AllChem.ETKDG())

    atom_symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    atom_positions = molecule.GetConformer().GetPositions()
    x_coordinates = [pos[0] for pos in atom_positions]
    y_coordinates = [pos[1] for pos in atom_positions]
    z_coordinates = [pos[2] for pos in atom_positions]
    radii = [atomic_radii.get(symbol, 1.0) for symbol in atom_symbols]

    # Get atom colors
    colors = [cpk_colors.get(symbol, "gray") for symbol in atom_symbols]

    def get_bonds() -> Dict[FrozenSet[int], float]:
        """Generates a set of bonds from the RDKit molecule

        Returns:
            Dict[FrozenSet[int], float]: A dictionary mapping pairs of atom indices to bond lengths.
        """
        bonds = dict()
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            dist = np.linalg.norm(
                np.array(atom_positions[i]) - np.array(atom_positions[j])
            )
            bonds[frozenset([i, j])] = dist
        return bonds

    def atom_trace() -> go.Scatter3d:
        """Creates an atom trace for the plot

        Returns:
            go.Scatter3d: The atom trace
        """
        # Use radii information to adjust atom sizes
        markers = dict(
            color=colors,
            line=dict(color="lightgray", width=2),
            size=[r * 10 for r in radii],  # Multiply by 10 for better visibility
            symbol="circle",
            opacity=0.8,
        )
        trace = go.Scatter3d(
            x=x_coordinates,
            y=y_coordinates,
            z=z_coordinates,
            mode="markers",
            marker=markers,
            text=atom_symbols,
            name="",
        )
        return trace

    def bond_trace() -> go.Scatter3d:
        """Creates a bond trace for the plot

        Returns:
            go.Scatter3d: The bond trace
        """
        trace = go.Scatter3d(
            x=[],
            y=[],
            z=[],
            hoverinfo="none",
            mode="lines",
            marker=dict(color="grey", size=7, opacity=1),
        )
        for i, j in bonds.keys():
            trace["x"] += (x_coordinates[i], x_coordinates[j], None)
            trace["y"] += (y_coordinates[i], y_coordinates[j], None)
            trace["z"] += (z_coordinates[i], z_coordinates[j], None)
        return trace

    # Get the bonds
    bonds = get_bonds()

    # Create annotations
    zipped = zip(range(len(atom_symbols)), x_coordinates, y_coordinates, z_coordinates)
    annotations_id = [
        dict(
            text=num, x=x, y=y, z=z, showarrow=False, yshift=15, font=dict(color="blue")
        )
        for num, x, y, z in zipped
    ]

    annotations_length = []
    for (i, j), dist in bonds.items():
        x_middle, y_middle, z_middle = (
            np.array(atom_positions[i]) + np.array(atom_positions[j])
        ) / 2
        annotation = dict(
            text=f"{dist:.2f}",
            x=x_middle,
            y=y_middle,
            z=z_middle,
            showarrow=False,
            yshift=15,
        )
        annotations_length.append(annotation)

    # Atom indices & Bond lengths
    annotations = annotations_id + annotations_length

    # Create the layout
    data = [atom_trace(), bond_trace()]
    axis_params = dict(
        showgrid=False,
        showbackground=False,
        showticklabels=False,
        zeroline=False,
        titlefont=dict(color="white"),
    )
    layout = dict(
        scene=dict(
            xaxis=axis_params,
            yaxis=axis_params,
            zaxis=axis_params,
            annotations=annotations,
        ),
        margin=dict(r=0, l=0, b=0, t=0),
        showlegend=False,
    )

    # Create the figure
    fig = go.Figure(data=data, layout=layout)
    return fig


def rotate_molecule_animation(
    figure: plotly.graph_objs.Figure,
    filedir: str,
    filename: str,
    duration: float = 1.0,
    frames: int = 30,
    rotations_per_sec: float = 1.0,
    overwrite: bool = False,
    engine: str = "kaleido",
) -> None:
    """Creates an animated GIF of the molecule rotating.

    Args:
        figure (plotly.graph_objs.Figure): The 3D plotly figure of the molecule.
        filedir (str): The directory to save the animated GIF.
        filename (str): The filename of the output animated GIF.
        duration (float, optional): Duration of the animation in seconds. Defaults to 1.0.
        frames (int, optional): Number of frames in the animation. Defaults to 30.
        rotations_per_sec (float, optional): Number of rotations per second. Defaults to 1.0.
        overwrite (bool, optional): If True, overwrite the file if it already exists. Defaults to False.
        engine (str, optional): engine to use for the .write_image plotly method. Defaults to "kaleido".
    """
    # Remove .gif extension if provided
    if filename.lower().endswith(".gif"):
        filename = filename[:-4]

    if not overwrite:
        # Check if the file already exists
        if os.path.isfile(os.path.join(filedir, f"{filename}.gif")):
            raise FileExistsError(
                f"{filename}.gif already exists. Set overwrite=True to overwrite the file."
            )

    # Create the animation frames
    animation_figures = []
    print("Creating the animation ...")
    for i in range(int(duration * frames)):
        layout = figure.layout
        layout["scene"]["camera"]["eye"] = dict(
            x=2 * np.sin(2 * np.pi * i * rotations_per_sec / frames),
            y=2 * np.cos(2 * np.pi * i * rotations_per_sec / frames),
            z=1,
        )
        fig = go.Figure(data=figure.data, layout=layout)
        animation_figures.append(fig)

    # Write the images to disk
    print("Saving the images ...")
    for i in tqdm(range(len(animation_figures))):
        fig = animation_figures[i]
        fig.write_image(f"{filename}_{i}.png", engine=engine)

    # Create the GIF
    print("Loading the images ...")
    images = []
    for i in tqdm(range(int(duration * frames))):
        images.append(imageio.imread(f"{filename}_{i}.png"))
    print("Creating the gif ...")
    imageio.imwrite(
        os.path.join(filedir, f"{filename}.gif"), images, duration=(1 / frames), loop=0
    )

    # Delete the images
    print("Deleting the images ...")
    for i in range(int(duration * frames)):
        if os.path.exists(f"{filename}_{i}.png"):
            os.remove(f"{filename}_{i}.png")


def plot_diffusion_trajectory(
    gen_obj: List[torch.Tensor], is_molecule: bool = False, dataset: str = "QM9"
) -> Union[plotly.graph_objs.Figure, matplotlib.figure.Figure]:
    """Return the figure of one generated object as part of a diffusion trajectory.

    Args:
        gen_obj (List[torch.Tensor]): The generated object (node features (x) and adjacency matrix (adj), and rank-2 incidence matrix (rank2) if we generated combinatorial complexes).
        is_molecule (bool, optional): if True, we plot a molecule, otherwise a graph. Defaults to False.
        dataset (str, optional): The dataset from which the object was generated. Defaults to "QM9" (only used if is_molecule=True).

    Returns:
        Union[plotly.graph_objs.Figure, matplotlib.figure.Figure]: The figure of the generated object.
    """
    x, adj = gen_obj[0], gen_obj[1]
    fig = plt.figure(figsize=(10, 10))
    if is_molecule:
        if dataset == "QM9":
            atomic_num_list = [6, 7, 8, 9, 0]
        else:
            atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        mol = construct_mol(x, adj, atomic_num_list)
        mol_img = Draw.MolToImage(mol, size=(300, 300))
        plt.imshow(mol_img)
        title_str = f"{Chem.MolToSmiles(mol)}"
    else:
        samples_int = quantize(adj.unsqueeze(0))
        G = adjs_to_graphs(samples_int, True)[0]
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)
        title_str = f"e={e - l}, n={v}"
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, **options)
    plt.title(title_str)
    plt.axis("off")
    return fig


def diffusion_animation(
    diff_traj: List[List[torch.Tensor]],
    is_molecule: bool = False,
    filedir: str = "./",
    filename: str = "diffusion_animation",
    fps: int = 25,
    overwrite: bool = True,
    engine: str = "kaleido",
) -> None:
    """Creates an animated GIF of the diffusion trajectory.

    Args:
        diff_traj (List[List[torch.Tensor]]): The diffusion trajectory (list of generated node features (x) and adjacency matrices (adj), and rank-2 incidence matrices (rank2) if we generated combinatorial complexes).
        is_molecule (bool, optional): If True, the frames are molecules not graphs. Defaults to False.
        filedir (str, optional): The directory to save the animated GIF. Defaults to "./".
        filename (str, optional): The filename of the output animated GIF. Defaults to "diffusion_animation".
        fps (int, optional): Number of frames per second. Defaults to 25.
        overwrite (bool, optional): If True, overwrite the file if it already exists. Defaults to True.
        engine (str, optional): engine to use for the .write_image plotly method if plotly is used. Defaults to "kaleido".
    """
    # Remove .gif extension if provided
    if filename.lower().endswith(".gif"):
        filename = filename[:-4]

    if not overwrite:
        # Check if the file already exists
        if os.path.isfile(os.path.join(filedir, f"{filename}.gif")):
            raise FileExistsError(
                f"{filename}.gif already exists. Set overwrite=True to overwrite the file."
            )

    # Create the animation frames
    animation_figures = []
    print("Creating the animation ...")
    for i in tqdm(range(len(diff_traj))):
        fig = plot_diffusion_trajectory(diff_traj[i], is_molecule=is_molecule)
        animation_figures.append(fig)

    # Write the images to disk
    print("Saving the images ...")
    for i in tqdm(range(len(animation_figures))):
        fig = animation_figures[i]
        if isinstance(fig, plotly.graph_objs.Figure):  # plotly
            fig.write_image(f"diffusion_{i}.png", engine=engine)
        elif isinstance(fig, matplotlib.figure.Figure):  # matplotlib
            fig.savefig(f"diffusion_{i}.png")
        else:
            raise TypeError(
                "The figure must be either a plotly.graph_objs.Figure or a matplotlib.figure.Figure. "
                "Otherwise, it has not been implemented yet."
            )

    # Create the GIF
    print("Loading the images ...")
    images = []
    for i in tqdm(range(len(animation_figures))):
        images.append(imageio.imread(f"diffusion_{i}.png"))
    print("Creating the gif ...")
    filepath = os.path.join(filedir, filename)
    imageio.imwrite(f"{filepath}.gif", images, duration=(1 / fps), loop=0)

    # Delete the images
    print("Deleting the images ...")
    for i in range(len(animation_figures)):
        if os.path.exists(f"diffusion_{i}.png"):
            os.remove(f"diffusion_{i}.png")

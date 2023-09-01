#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""data_generators.py: functions and GraphGenerator class for generating graphs and graph/combinatorial complexes datasets with given properties.
Run this script with -h flag to see usage on how to generate graph and combinatorial complex datasets.
The arguments are (see ccsd/src/parsers/parser_generator.py for more details):
    --data-dir: directory to save generated graphs. Default: "data".
    --dataset: name of dataset to generate (default "grid"), choices are ["ego_small", "community_small", "ENZYMES", "ENZYMES_small", "grid"].
    --is_cc: if you want to generate combinatorial complexes instead of graphs
    --folder: Directory to save the results, load checkpoints, load config, etc. Default: "./".
Adapted from Jo, J. & al (2022) for the graph generation part.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import scipy.sparse as sp
from toponetx.classes.combinatorial_complex import CombinatorialComplex

sys.path.insert(0, os.getcwd())

from ccsd.src.parsers.parser_generator import ParserGenerator
from ccsd.src.utils.cc_utils import convert_graphs_to_CCs


def n_community(
    num_communities: int, max_nodes: int, p_inter: float = 0.05
) -> nx.Graph:
    """Generate a graph with `num_communities` communities, each of size `max_nodes` and with inter-community edge probability `p_inter`.
    From Niu et al. (2020)

    Args:
        num_communities (int): number of communities
        max_nodes (int): maximum number of nodes in each community
        p_inter (float, optional): inter-community edge probability. Defaults to 0.05.

    Returns:
        nx.Graph: generated graph
    """
    assert num_communities > 1

    one_community_size = max_nodes // num_communities
    c_sizes = [one_community_size] * num_communities
    total_nodes = one_community_size * num_communities
    p_make_a_bridge = p_inter * 2 / ((num_communities - 1) * one_community_size)

    print(num_communities, total_nodes, end=" ")
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]

    G = nx.disjoint_union_all(graphs)
    communities = list(G.subgraph(c) for c in nx.connected_components(G))
    add_edge = 0
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i + 1, len(communities)):  # loop for C_M^2 times
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:  # loop for N times
                for n2 in nodes2:  # loop for N times
                    if np.random.rand() < p_make_a_bridge:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
                        add_edge += 1
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
                add_edge += 1
    print(
        "connected comp: ",
        len(list(G.subgraph(c) for c in nx.connected_components(G))),
        "add edges: ",
        add_edge,
    )
    print(G.number_of_edges())
    return G


# Dictionary of graph generators
NAME_TO_NX_GENERATOR = {
    "community": n_community,
    "grid": nx.generators.grid_2d_graph,
    "gnp": nx.generators.fast_gnp_random_graph,  # fast_gnp_random_graph(n, p, seed=None, directed=False)
    "ba": nx.generators.barabasi_albert_graph,  # barabasi_albert_graph(n, m, seed=None)
    "pow_law": lambda **kwargs: nx.configuration_model(
        nx.generators.random_powerlaw_tree_sequence(**kwargs, gamma=3, tries=2000)
    ),
    "except_deg": lambda **kwargs: nx.expected_degree_graph(**kwargs, selfloops=False),
    "cycle": nx.cycle_graph,
    "c_l": nx.circular_ladder_graph,
    "lobster": nx.random_lobster,
}


class GraphGenerator:
    """Graph generator class."""

    def __init__(
        self,
        graph_type: str = "grid",
        possible_params_dict: Optional[Dict[str, Union[int, np.ndarray]]] = None,
        corrupt_func: Optional[Callable[[Any], nx.Graph]] = None,
    ) -> None:
        """Initialize graph generator.

        Args:
            graph_type (str, optional): type of graphs to generate. Defaults to "grid".
            possible_params_dict (Optional[Dict[str, Union[int, np.ndarray]]], optional): set of parameters to randomly select. Defaults to None.
            corrupt_func (Optional[Callable[[Any], nx.Graph]], optional): optional function that generates a constant graph (for debugging for example). Defaults to None.
        """
        if possible_params_dict is None:
            possible_params_dict = {}
        assert isinstance(possible_params_dict, dict)
        self.count = {k: 0 for k in possible_params_dict}
        self.possible_params = possible_params_dict
        self.corrupt_func = corrupt_func
        self.nx_generator = NAME_TO_NX_GENERATOR[graph_type]

    def __call__(self) -> nx.Graph:
        """Generate a graph.

        Returns:
            nx.Graph: generated graph
        """
        params = {}
        # Randomly choose parameters from possible_params
        for k, v_list in self.possible_params.items():
            params[k] = np.random.choice(v_list)
        # Generate graph
        graph = self.nx_generator(**params)
        # Relabel nodes to be 0, 1, 2, ...
        graph = nx.relabel.convert_node_labels_to_integers(
            graph
        )  # convert to consecutive integers
        if self.corrupt_func is not None:  # Corrupt graph
            graph = self.corrupt_func(self.corrupt_func)
        return graph

    def __repr__(self) -> str:
        """Return string representation of graph generator.

        Returns:
            str: string representation of graph generator
        """
        return f"GraphGenerator({self.possible_params})"


def gen_graph_list(
    graph_type: str = "grid",
    possible_params_dict: Optional[Dict[str, Union[int, np.ndarray]]] = None,
    corrupt_func: Optional[Callable[[Any], nx.Graph]] = None,
    length: int = 1024,
    save_dir: Optional[str] = None,
    file_name: Optional[str] = None,
    max_node: Optional[int] = None,
    min_node: Optional[int] = None,
) -> List[nx.Graph]:
    """Generate a list of synthetic graphs.

    Args:
        graph_type (str, optional): type of graphs to generate. Defaults to "grid".
        possible_params_dict (Optional[Dict[str, Union[int, np.ndarray]]], optional): set of parameters to randomly select. Defaults to None.
        corrupt_func (Optional[Callable[[Any], nx.Graph]], optional): optional function that generates a constant graph (for debugging for example). Defaults to None.
        length (int, optional): number of graphs to generate. Defaults to 1024.
        save_dir (Optional[str], optional): where to save the generate list of graph. Defaults to None.
        file_name (Optional[str], optional): name of the file. Defaults to None.
        max_node (Optional[int], optional): maximum number of nodes. Defaults to None.
        min_node (Optional[int], optional): minimum number of nodes. Defaults to None.

    Returns:
        List[nx.Graph]: list of generated graphs
    """
    params = locals()
    logging.info("gen data: " + json.dumps(params))
    if file_name is None:
        file_name = graph_type + "_" + str(length)
    graph_generator = GraphGenerator(
        graph_type=graph_type,
        possible_params_dict=possible_params_dict,
        corrupt_func=corrupt_func,
    )
    graph_list = []
    i = 0
    max_N = 0
    while i < length:
        graph = graph_generator()
        if max_node is not None and graph.number_of_nodes() > max_node:
            continue
        if min_node is not None and graph.number_of_nodes() < min_node:
            continue
        print(i, graph.number_of_nodes(), graph.number_of_edges())
        max_N = max(max_N, graph.number_of_nodes())
        if graph.number_of_nodes() <= 1:
            continue
        graph_list.append(nx.convert_node_labels_to_integers(graph))
        i += 1
    if save_dir is not None:
        file_path = os.path.join(save_dir, file_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(file_path + ".pkl", "wb") as f:
            pickle.dump(obj=graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(file_path + ".txt", "w") as f:
            f.write(json.dumps(params))
            f.write(f"max node number: {max_N}")
    print(max_N)
    return graph_list


def load_dataset(
    data_dir: str = "data", file_name: Optional[str] = None
) -> Union[List[nx.Graph], List[CombinatorialComplex]]:
    """Load an existing dataset as a list of graphs or list of combinatorial complexes from a file.

    Args:
        data_dir (str, optional): directory of the dataset. Defaults to "data".
        file_name (Optional[str], optional): name of the file. Defaults to None.

    Returns:
        Union[List[nx.Graph], List[CombinatorialComplex]]: list of graphs or list of combinatorial complexes
    """
    file_path = os.path.join(data_dir, file_name)
    with open(file_path + ".pkl", "rb") as f:
        obj_list = pickle.load(f)
    return obj_list


def graph_load_batch(
    min_num_nodes: int = 20,
    max_num_nodes: int = 1000,
    name: str = "ENZYMES",
    node_attributes: bool = True,
    graph_labels: bool = True,
    folder: str = "./",
) -> List[nx.Graph]:
    """Load a graph dataset, for ENZYMES, PROTEIN and DD.

    Args:
        min_num_nodes (int, optional): minimum number of nodes. Defaults to 20.
        max_num_nodes (int, optional): maximum number of nodes. Defaults to 1000.
        name (str, optional): name of the dataset to load. Defaults to "ENZYMES".
        node_attributes (bool, optional): if True, also load the node attributes. Defaults to True.
        graph_labels (bool, optional): if True, also load the graph labels. Defaults to True.
        folder (str, optional): directory of the data/dataset/ folders. Defaults to "./".

    Returns:
        List[nx.Graph]: list of graphs
    """
    print("Loading graph dataset: " + str(name))
    G = nx.Graph()  # start with an empty graph
    # Load the data
    path = os.path.join(folder, "data", "dataset", name)
    data_adj = np.loadtxt(os.path.join(path, f"{name}_A.txt"), delimiter=",").astype(
        int
    )
    data_node_att = []
    if node_attributes:  # Load the node attributes
        data_node_att = np.loadtxt(
            os.path.join(path, f"{name}_node_attributes.txt"), delimiter=","
        )
    data_node_label = np.loadtxt(
        os.path.join(path, f"{name}_node_labels.txt"), delimiter=","
    ).astype(int)
    data_graph_indicator = np.loadtxt(
        os.path.join(path, f"{name}_graph_indicator.txt"), delimiter=","
    ).astype(int)
    if graph_labels:  # Load the graph labels
        data_graph_labels = np.loadtxt(
            os.path.join(path, f"{name}_graph_labels.txt"), delimiter=","
        ).astype(int)

    data_tuple = list(map(tuple, data_adj))  # convert to tuple

    # Add edges to the graph
    G.add_edges_from(data_tuple)

    # Add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
        G.add_node(i + 1, label=data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    print(G.number_of_nodes())
    print(G.number_of_edges())

    # Split into multiple graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # Find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)  # extract the subgraph
        if graph_labels:
            G_sub.graph["label"] = data_graph_labels[i]
        # Check if the number of nodes in this graph is in the specified range
        if (min_num_nodes <= G_sub.number_of_nodes()) and (
            G_sub.number_of_nodes() <= max_num_nodes
        ):
            # Relabel the nodes using consecutive integers
            graphs.append(nx.convert_node_labels_to_integers(G_sub))
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
    print(f"Graphs loaded, total num: {len(graphs)}")
    return graphs


def parse_index_file(filename: str) -> List[int]:
    """Parse an index file (list of integers).

    Args:
        filename (str): name of the file

    Returns:
        List[int]: list of indices as integers
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def graph_load(
    dataset: str = "cora", folder: str = "./"
) -> Tuple[sp.spmatrix, List[nx.Graph]]:
    """Load the citation datasets: cora, citeseer or pubmed.

    Args:
        dataset (str, optional): name of the dataset to load. Defaults to "cora".
        folder (str, optional): directory of the data/dataset/ folders. Defaults to "./".

    Returns:
        Tuple[sp.spmatrix, List[nx.Graph]]: tuple of features and the graph
    """
    names = ["x", "tx", "allx", "graph"]
    objects = []
    for i in range(len(names)):
        load = pickle.load(
            open(
                os.path.join(
                    folder, "data", "dataset", "ind.{}.{}".format(dataset, names[i])
                ),
                "rb",
            ),
            encoding="latin1",
        )
        objects.append(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        os.path.join(folder, "data", "dataset", "ind.{}.test.index".format(dataset))
    )
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == "citeseer":  # Special case for citeseer
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    return features, G


def citeseer_ego(
    radius: int = 3, node_min: int = 50, node_max: int = 400, folder: str = "./"
) -> List[nx.Graph]:
    """Load the citeseer dataset, keep the largest connected component, and extract the ego graphs
    (graphs of nodes within a certain radius) with a number of nodes within our range.

    Args:
        radius (int, optional): radius. Defaults to 3.
        node_min (int, optional): minimum number of nodes in our dataset. Defaults to 50.
        node_max (int, optional): maximum number of nodes in our dataset. Defaults to 400.
        folder (str, optional): directory of the data/dataset/ folders. Defaults to "./".

    Returns:
        List[nx.Graph]: list of (ego) graphs
    """
    # Load citeseer
    _, G = graph_load(dataset="citeseer", folder=folder)
    # Keep the largest connected component
    G = max([G.subgraph(c) for c in nx.connected_components(G)], key=len)
    # Relabel the nodes using consecutive integers
    G = nx.convert_node_labels_to_integers(G)
    graphs = []
    for i in range(G.number_of_nodes()):
        G_ego = nx.ego_graph(G, i, radius=radius)
        assert isinstance(G_ego, nx.Graph)
        # Check if the number of nodes in this graph is in the specified range
        if (node_min <= G_ego.number_of_nodes()) and (
            G_ego.number_of_nodes() <= node_max
        ):
            G_ego.remove_edges_from(nx.selfloop_edges(G_ego))
            graphs.append(
                nx.convert_node_labels_to_integers(G_ego)
            )  # convert to consecutive integers
    return graphs


def save_dataset(
    data_dir: str,
    obj: Union[List[nx.Graph], List[CombinatorialComplex]],
    save_name: str,
    save_txt: bool = True,
) -> None:
    """Save the dataset (objects) in the specified directory.

    Args:
        data_dir (str): directory to save the dataset
        obj (Union[List[nx.Graph], List[CombinatorialComplex]]): list of objects to save
        save_name (str): name of the dataset
        save_txt (bool, optional): whether to save a txt file with the name and the number of objects (or size of DataLoader). Defaults to True.
    """
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    file_path = os.path.join(data_dir, save_name)
    print(save_name, len(obj))
    with open(file_path + ".pkl", "wb") as f:
        pickle.dump(obj=obj, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    if save_txt:
        with open(file_path + ".txt", "w") as f:
            f.write(save_name + "\n")
            f.write(str(len(obj)))


def generate_dataset(args: argparse.Namespace) -> None:
    """Generate a graph/combinatorial complex dataset and save it in the specified directory.

    Args:
        args (argparse.Namespace): arguments

    Raises:
        NotImplementedError: raise and error if the specified dataset is not implemented
    """
    data_dir = os.path.join(args.folder, args.data_dir)  # default: "./data/"
    dataset = args.dataset  # default: "community_small"
    is_cc = args.is_cc  # default: False

    if dataset == "community_small":
        # Generate 100 community graphs with 2 communities and 12-20 nodes
        # (dataset already save within the function)
        graphs = gen_graph_list(
            graph_type="community",
            possible_params_dict={
                "num_communities": [2],
                "max_nodes": np.arange(12, 21).tolist(),
            },
            corrupt_func=None,
            length=100,
        )
        max_nb_nodes = max([g.number_of_nodes() for g in graphs])
        if not (is_cc):
            save_dataset(data_dir, graphs, dataset)
        else:
            ccs = convert_graphs_to_CCs(
                graphs,
                is_molecule=False,
                lifting_procedure="path_based",
                lifting_procedure_kwargs={
                    "sources_nodes": list(range(max_nb_nodes)),
                    "path_length": 3,
                },
                max_nb_nodes=max_nb_nodes,
            )
            save_dataset(data_dir, ccs, f"{dataset}_CC")
        print(max_nb_nodes)

    elif dataset == "grid":
        # Generate 100 grid graphs with 10-19 rows and 10-19 columns
        # (dataset already save within the function)
        graphs = gen_graph_list(
            graph_type="grid",
            possible_params_dict={
                "m": np.arange(10, 20).tolist(),
                "n": np.arange(10, 20).tolist(),
            },
            corrupt_func=None,
            length=100,
        )
        max_nb_nodes = max([g.number_of_nodes() for g in graphs])
        if not (is_cc):
            save_dataset(data_dir, graphs, dataset)
        else:
            ccs = convert_graphs_to_CCs(
                graphs,
                is_molecule=False,
                lifting_procedure="path_based",
                lifting_procedure_kwargs={
                    "sources_nodes": list(range(max_nb_nodes)),
                    "path_length": 3,
                },
                max_nb_nodes=max_nb_nodes,
            )
            save_dataset(data_dir, ccs, f"{dataset}_CC")
        print(max_nb_nodes)

    elif dataset == "grid_small":
        # Generate 100 grid graphs with 4-7 rows and 4-7 columns
        # (dataset already save within the function)
        # smaller dataset than the original grid dataset used in benchmarks
        graphs = gen_graph_list(
            graph_type="grid",
            possible_params_dict={
                "m": np.arange(4, 8).tolist(),
                "n": np.arange(4, 8).tolist(),
            },
            corrupt_func=None,
            length=100,
        )
        max_nb_nodes = max([g.number_of_nodes() for g in graphs])
        if not (is_cc):
            save_dataset(data_dir, graphs, dataset)
        else:
            ccs = convert_graphs_to_CCs(
                graphs,
                is_molecule=False,
                lifting_procedure="path_based",
                lifting_procedure_kwargs={
                    "sources_nodes": list(range(max_nb_nodes)),
                    "path_length": 3,
                },
            )
            save_dataset(data_dir, ccs, f"{dataset}_CC")
        print(max_nb_nodes)

    elif dataset == "ego_small":
        # Generate 200 ego graphs from the citeseer dataset with radius 1 and 4-18 nodes
        graphs = citeseer_ego(radius=1, node_min=4, node_max=18, folder=args.folder)[
            :200
        ]
        if is_cc:
            ccs = convert_graphs_to_CCs(
                graphs,
                is_molecule=False,
                lifting_procedure="cycles",
                lifting_procedure_kwargs=None,
            )
            save_dataset(data_dir, ccs, f"{dataset}_CC")
        else:
            save_dataset(data_dir, graphs, dataset)
        print(max([g.number_of_nodes() for g in graphs]))

    elif dataset == "ENZYMES":
        # Load and save the ENZYMES dataset (graphs with 10-1000 nodes)
        # Don't keep the node attributes but keep the graph labels
        graphs = graph_load_batch(
            min_num_nodes=10,
            max_num_nodes=1000,
            name=dataset,
            node_attributes=False,
            graph_labels=True,
            folder=args.folder,
        )
        if not (is_cc):
            save_dataset(data_dir, graphs, dataset)
        else:
            ccs = convert_graphs_to_CCs(
                graphs, is_molecule=False, lifting_procedure="cycles"
            )
            save_dataset(data_dir, ccs, f"{dataset}_CC")
        print(max([g.number_of_nodes() for g in graphs]))

    elif dataset == "ENZYMES_small":
        # Load and save the ENZYMES dataset, but a smaller version (graphs with 1-12 nodes only)
        # Don't keep the node attributes but keep the graph labels
        graphs = graph_load_batch(
            min_num_nodes=1,
            max_num_nodes=12,
            name="ENZYMES",
            node_attributes=False,
            graph_labels=True,
            folder=args.folder,
        )
        if not (is_cc):
            save_dataset(data_dir, graphs, dataset)
        else:
            ccs = convert_graphs_to_CCs(
                graphs, is_molecule=False, lifting_procedure="cycles"
            )
            save_dataset(data_dir, ccs, f"{dataset}_CC")
        print(max([g.number_of_nodes() for g in graphs]))

    else:
        # Dataset not supported
        raise NotImplementedError(f"Dataset {dataset} not supported.")


if __name__ == "__main__":
    # Parse arguments
    args = ParserGenerator().parse()
    # Generate and save the dataset
    generate_dataset(args)

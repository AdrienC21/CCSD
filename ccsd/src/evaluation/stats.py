#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""stats.py: code for computing statistics of graphs.

Adapted from Jo, J. & al (2022)
"""

import concurrent.futures
import os
import random
import subprocess as sp
from datetime import datetime
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from easydict import EasyDict
from scipy.linalg import eigvalsh

from ccsd.src.evaluation.mmd import (
    compute_mmd,
    compute_nspdk_mmd,
    gaussian,
    gaussian_emd,
    process_tensor,
)
from ccsd.src.utils.graph_utils import adjs_to_graphs

# Global variables
PRINT_TIME = False  # whether to print the time for computing statistics


def degree_worker(G: nx.Graph) -> np.ndarray:
    """Function for computing the degree histogram of a graph.

    Returns:
        np.ndarray: degree histogram
    """
    return np.array(nx.degree_histogram(G))


def add_tensor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Function for extending the dimension of two tensors to
    make them having the same support and add them together.

    Args:
        x (np.ndarray): vector 1
        y (np.ndarray): vector 2

    Returns:
        np.ndarray: sum of vector 1 and vector 2
    """
    x, y = process_tensor(x, y)
    return x + y


def degree_stats(
    graph_ref_list: List[nx.Graph],
    graph_pred_list: List[nx.Graph],
    kernel: Callable[[np.ndarray, np.ndarray], float] = gaussian_emd,
    is_parallel: bool = True,
    max_workers: Optional[int] = None,
    debug_mode: bool = False,
) -> float:
    """Compute the MMD distance between the degree distributions of two unordered sets of graphs.

    Args:
        graph_ref_list (List[nx.Graph]): reference list of networkx graphs to be evaluated
        graph_pred_list (List[nx.Graph]): target list of networkx graphs to be evaluated
        kernel (Callable[[np.ndarray, np.ndarray], float], optional): kernel function. Defaults to gaussian_emd.
        is_parallel (bool, optional): if True, do parallel computing. Defaults to True.
        max_workers (Optional[int], optional): number of workers (if is_parallel). Defaults to None.
        debug_mode (bool, optional): whether or not we print debug info for parallel computing. Defaults to False.

    Returns:
        float: MMD distance
    """

    sample_ref = []
    sample_pred = []
    # Remove empty graphs if generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        if debug_mode:
            print("Start parallel computing for degree mmd reference objects")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(degree_worker, graph_ref_list)
            try:
                for deg_hist in results:
                    sample_ref.append(deg_hist)
            except Exception as e:
                raise e
        if debug_mode:
            print("Start parallel computing for degree mmd predicted objects")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(degree_worker, graph_pred_list_remove_empty)
            try:
                for deg_hist in results:
                    sample_pred.append(deg_hist)
            except Exception as e:
                raise e

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    # Compute MMD
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=kernel)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


def spectral_worker(G: nx.Graph) -> np.ndarray:
    """Function for computing the spectral density of a graph.

    Args:
        G (nx.Graph): input graph

    Returns:
        np.ndarray: spectral density
    """
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def spectral_stats(
    graph_ref_list: List[nx.Graph],
    graph_pred_list: List[nx.Graph],
    kernel: Callable[[np.ndarray, np.ndarray], float] = gaussian_emd,
    is_parallel: bool = True,
    max_workers: Optional[int] = None,
    debug_mode: bool = False,
) -> np.ndarray:
    """Compute the MMD distance between the spectral densities of two unordered sets of graphs.

    Args:
        graph_ref_list (List[nx.Graph]): reference list of networkx graphs to be evaluated
        graph_pred_list (List[nx.Graph]): target list of networkx graphs to be evaluated
        kernel (Callable[[np.ndarray, np.ndarray], float], optional): kernel function. Defaults to gaussian_emd.
        is_parallel (bool, optional): if True, do parallel computing. Defaults to True.
        max_workers (Optional[int], optional): number of workers (if is_parallel). Defaults to None.
        debug_mode (bool, optional): whether or not we print debug info for parallel computing. Defaults to False.

    Returns:
        np.ndarray: spectral distance
    """

    sample_ref = []
    sample_pred = []

    # Remove empty graphs if generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        if debug_mode:
            print("Start parallel computing for spectral mmd reference objects")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(spectral_worker, graph_ref_list)
            try:
                for spectral_density in results:
                    sample_ref.append(spectral_density)
            except Exception as e:
                raise e
        if debug_mode:
            print("Start parallel computing for spectral mmd predicted objects")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(spectral_worker, graph_pred_list_remove_empty)
            try:
                for spectral_density in results:
                    sample_pred.append(spectral_density)
            except Exception as e:
                raise e
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=kernel)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


def clustering_worker(param: Tuple[nx.Graph, int]) -> np.ndarray:
    """Function for computing the histogram of clustering coefficient of a graph.

    Args:
        param (Tuple[nx.Graph, int]): input graph and number of bins

    Returns:
        np.ndarray: histogram of clustering coefficient
    """
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
    )
    return hist


def clustering_stats(
    graph_ref_list: List[nx.Graph],
    graph_pred_list: List[nx.Graph],
    kernel: Callable[[np.ndarray, np.ndarray], float] = gaussian_emd,
    bins: int = 100,
    is_parallel: bool = True,
    max_workers: Optional[int] = None,
    debug_mode: bool = False,
) -> np.ndarray:
    """Compute the MMD distance between the clustering coefficients of two unordered sets of graphs.
    For unweighted graphs, the clustering coefficient of a node u is the fraction of possible triangles through that node that exist.


    Args:
        graph_ref_list (List[nx.Graph]): reference list of networkx graphs to be evaluated
        graph_pred_list (List[nx.Graph]): target list of networkx graphs to be evaluated
        kernel (Callable[[np.ndarray, np.ndarray], float], optional): kernel function. Defaults to gaussian_emd.
        bins (int, optional): number of bins for the histogram. Defaults to 100.
        is_parallel (bool, optional): if True, do parallel computing. Defaults to True.
        max_workers (Optional[int], optional): number of workers (if is_parallel). Defaults to None.
        debug_mode (bool, optional): whether or not we print debug info for parallel computing. Defaults to False.

    Returns:
        float: mmd distance
    """

    sample_ref = []
    sample_pred = []

    # Remove empty graphs if generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        if debug_mode:
            print("Start parallel computing for clustering mmd reference objects")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(
                clustering_worker, [(G, bins) for G in graph_ref_list]
            )
            try:
                for clustering_hist in results:
                    sample_ref.append(clustering_hist)
            except Exception as e:
                raise e
        if debug_mode:
            print("Start parallel computing for clustering mmd predicted objects")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(
                clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]
            )
            try:
                for clustering_hist in results:
                    sample_pred.append(clustering_hist)
            except Exception as e:
                raise e
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values()
            )
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)
    try:
        mmd_dist = compute_mmd(
            sample_ref,
            sample_pred,
            kernel=kernel,
            sigma=1.0 / 10,
            distance_scaling=bins,
        )
    except:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=kernel, sigma=1.0 / 10)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing clustering mmd: ", elapsed)
    return mmd_dist


# Maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    "3path": [1, 2],
    "4cycle": [8],
}
# Format of the start of the orbit counts in orca output
COUNT_START_STR = "orbit counts: \n"


def edge_list_reindexed(G: nx.Graph) -> List[Tuple[int, int]]:
    """Reindex the nodes of a graph to be contiguous integers starting from 0.

    Args:
        G (nx.Graph): input graph

    Returns:
        List[Tuple[int, int]]: list of edges (index_u, index_v)
    """
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for u, v in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph: nx.Graph, orca_dir: str) -> np.ndarray:
    """Compute the orbit counts of a graph using orca.

    Args:
        graph (nx.Graph): input graph
        orca_dir (str): path to the orca directory where the executable are

    Returns:
        np.ndarray: orbit counts
    """
    tmp_file_path = os.path.join(orca_dir, f"tmp-{random.random():.4f}.txt")
    f = open(tmp_file_path, "w")
    f.write(str(graph.number_of_nodes()) + " " + str(graph.number_of_edges()) + "\n")
    for u, v in edge_list_reindexed(graph):
        f.write(str(u) + " " + str(v) + "\n")
    f.close()

    output = sp.check_output(
        [os.path.join(orca_dir, "orca"), "node", "4", tmp_file_path, "std"]
    )
    output = output.decode("utf8").strip()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array(
        [
            list(map(int, node_cnts.strip().split(" ")))
            for node_cnts in output.strip("\n").split("\n")
        ]
    )

    try:
        os.remove(tmp_file_path)
    except OSError:
        pass

    return node_orbit_counts


def orbit_stats_all(
    graph_ref_list: List[nx.Graph],
    graph_pred_list: List[nx.Graph],
    kernel: Callable[[np.ndarray, np.ndarray], float] = gaussian,
    folder: str = "./",
) -> float:
    """Compute the MMD distance between the orbits of two unordered sets of graphs.

    Args:
        graph_ref_list (List[nx.Graph]): reference list of networkx graphs to be evaluated
        graph_pred_list (List[nx.Graph]): target list of networkx graphs to be evaluated
        kernel (Callable[[np.ndarray, np.ndarray], float], optional): kernel function. Defaults to gaussian.
        folder (str, optional): path to the main folder where the ccsd/src/evaluation folders are to locate the orca executable. Defaults to "./".

    Returns:
        float: mmd distance
    """
    total_counts_ref = []
    total_counts_pred = []

    prev = datetime.now()

    orca_dir = os.path.join(
        *[folder, "ccsd", "src", "evaluation", "orca"]
    )  # path to the orca dir

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G, orca_dir)
        except Exception as e:
            print(e)
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G, orca_dir)
        except:
            print("orca failed")
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = compute_mmd(
        total_counts_ref, total_counts_pred, kernel=kernel, is_hist=False, sigma=30.0
    )

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing orbit mmd: ", elapsed)
    return mmd_dist


def nspdk_stats(
    graph_ref_list: List[nx.Graph], graph_pred_list: List[nx.Graph]
) -> float:
    """Compute the MMD distance between the NSPDK kernel of two unordered sets of graphs.

    Adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/stats.py

    Args:
        graph_ref_list (List[nx.Graph]): reference list of networkx graphs to be evaluated
        graph_pred_list (nx.Graph): target list of networkx graphs to be evaluated

    Returns:
        float: mmd distance
    """
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    mmd_dist = compute_nspdk_mmd(
        graph_ref_list,
        graph_pred_list_remove_empty,
        metric="nspdk",
        is_hist=False,
        n_jobs=20,
    )
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


# Dictionary mapping method names to functions to compute different MMD distances
METHOD_NAME_TO_FUNC = {
    "degree": degree_stats,
    "cluster": clustering_stats,
    "orbit": orbit_stats_all,
    "spectral": spectral_stats,
    "nspdk": nspdk_stats,
}


def eval_graph_list(
    graph_ref_list: List[nx.Graph],
    graph_pred_list: List[nx.Graph],
    methods: Optional[List[str]] = None,
    kernels: Optional[
        Dict[
            str,
            Callable[[np.ndarray, np.ndarray], float],
        ]
    ] = None,
    folder: str = "./",
) -> Dict[str, float]:
    """Evaluate generated generic graphs against a reference set of graphs using a set of methods and their corresponding kernels.

    Args:
        graph_ref_list (List[nx.Graph]): reference list of networkx graphs to be evaluated
        graph_pred_list (List[nx.Graph]): target list of networkx graphs to be evaluated
        methods (Optional[List[str]], optional): methods to be evaluated. Defaults to None.
        kernels (Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]], optional): kernels to be used for each methods. Defaults to None.
        folder (str, optional): path to the main folder where the ccsd/src/evaluation folders are to locate the orca executable. Defaults to "./".

    Returns:
        Dict[str, float]: dictionary mapping method names to their corresponding scores
    """
    if (
        methods is None
    ):  # by default, evaluate the methods ["degree", "cluster", "orbit"]
        methods = ["degree", "cluster", "orbit"]
    results = {}
    for method_id, method in enumerate(methods):
        print(f"Evaluating method: {method} ({method_id+1}/{len(methods)}) ...")
        top = perf_counter()
        if (
            method == "nspdk"
        ):  # nspdk requires a different function signature as there is no kernel as input
            results[method] = METHOD_NAME_TO_FUNC[method](
                graph_ref_list, graph_pred_list
            )
        elif (
            method == "orbit"
        ):  # orbit requires a different function signature with the folder provided
            results[method] = round(
                METHOD_NAME_TO_FUNC[method](
                    graph_ref_list, graph_pred_list, kernels[method], folder=folder
                ),
                6,
            )
        else:
            results[method] = round(
                METHOD_NAME_TO_FUNC[method](
                    graph_ref_list, graph_pred_list, kernels[method]
                ),
                6,
            )
        print(
            "\033[91m"
            + f"{method:9s}"
            + "\033[0m"
            + " : "
            + "\033[94m"
            + f"{results[method]:.6f}"
            + "\033[0m"
        )
        print(f"Time elapsed: {round(perf_counter() - top, 3)}s")
    return results


def eval_torch_batch(
    ref_batch: torch.Tensor,
    pred_batch: torch.Tensor,
    methods: Optional[List[str]] = None,
    folder: str = "./",
) -> Dict[str, float]:
    """Evaluate generated generic graphs against a reference set of graphs using a set of methods and their corresponding kernels,
    with the input graphs in torch.Tensor format (adjacency matrices).

    Args:
        ref_batch (torch.Tensor): reference batch of adjacency matrices
        pred_batch (torch.Tensor): target batch of adjacency matrices
        methods (Optional[List[str]], optional): methods to be evaluated. Defaults to None.
        folder (str, optional): path to the main folder where the ccsd/src/evaluation folders are to locate the orca executable. Defaults to "./".

    Returns:
        Dict[str, float]: dictionary mapping method names to their corresponding scores
    """
    graph_ref_list = adjs_to_graphs(ref_batch.detach().cpu().numpy())
    graph_pred_list = adjs_to_graphs(pred_batch.detach().cpu().numpy())
    results = eval_graph_list(
        graph_ref_list, graph_pred_list, methods=methods, folder=folder
    )
    return results

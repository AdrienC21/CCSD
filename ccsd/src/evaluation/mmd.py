#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""mmd.py: code for computing MMD (Maximum Mean Discrepancy),
kernel based statistical test used to determine whether given two
distribution are the same. Also contains functions to calculate
the EMD (Earth Mover's Distance) and the L2 distance between two
histograms, in addition to Gaussian kernels with these distances.

Adapted from Jo, J. & al (2022)
"""

import concurrent.futures
from functools import partial
from typing import Callable, Iterator, List, Optional, Tuple

import networkx as nx
import numpy as np
import pyemd
from scipy.linalg import toeplitz
from sklearn.metrics.pairwise import pairwise_kernels

from .eden import vectorize


def emd(x: np.ndarray, y: np.ndarray, distance_scaling: float = 1.0) -> float:
    """
    Calculate the earth mover's distance (EMD) between two histograms
    It corresponds to the Wasserstein metric (see Optimal transport theory)
    The formula is (\inf_{\gama \in \Gama(\mu, \nu) \int_{M*M} d(x,y)^p d\gama(x,y))^(1/p).

    Adapted from From Niu et al. (2020)

    Args:
        x (np.ndarray): histogram of first distribution
        y (np.ndarray): histogram of second distribution
        distance_scaling (float, optional): distance scaling factor. Defaults to 1.0.

    Returns:
        float: EMD value
    """

    # Convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    support_size = max(len(x), len(y))  # support of the two vectors
    # Diagonal-constant matrix
    d_mat = toeplitz(range(support_size)).astype(np.float64)
    distance_mat = d_mat / distance_scaling
    x, y = process_tensor(x, y)
    # Calculate EMD
    emd_value = pyemd.emd(x, y, distance_mat)
    return np.abs(emd_value)


def l2(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the L2 distance between two histograms

    Args:
        x (np.ndarray): histogram of first distribution
        y (np.ndarray): histogram of second distribution

    Returns:
        float: L2 distance
    """
    dist = np.linalg.norm(x - y, 2)
    return dist


def gaussian_emd(
    x: np.ndarray, y: np.ndarray, sigma: float = 1.0, distance_scaling: float = 1.0
) -> float:
    """Gaussian kernel with squared distance in exponential term replaced by EMD
    The inputs are PMF (Probability mass function). The Gaussian kernel is defined as
    k(x,y) = exp(-f(x,y)^2/(2*sigma^2)) where f(.,.) is the EMD function.

    Args:
        x (np.ndarray): 1D pmf of the first distribution with the same support
        y (np.ndarray): 1D pmf of the second distribution with the same support
        sigma (float, optional): standard deviation. Defaults to 1.0.
        distance_scaling (float, optional): distance scaling factor. Defaults to 1.0.

    Returns:
        float: Gaussian kernel value
    """
    emd_value = emd(x, y, distance_scaling)
    return np.exp(-emd_value * emd_value / (2 * sigma * sigma))


def gaussian(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Gaussian kernel with squared distance in exponential term replaced by L2 distance
    The inputs are PMF (Probability mass function). The Gaussian kernel is defined as
    k(x,y) = exp(-||x - y||^2/(2*sigma^2)) where ||.|| is the L2 distance function.

    Args:
        x (np.ndarray): 1D pmf of the first distribution with the same support
        y (np.ndarray): 1D pmf of the second distribution with the same support
        sigma (float, optional): standard deviation. Defaults to 1.0.

    Returns:
        float: Gaussian kernel value
    """

    # Convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x, y = process_tensor(x, y)
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def gaussian_tv(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Gaussian kernel with squared distance in exponential term replaced by total variation distance (half L1 distance, used in transportation theory)
    The inputs are PMF (Probability mass function). The Gaussian kernel is defined as
    k(x,y) = exp(-f(x - y)^2/(2*sigma^2)) where f(.) = 0.5 * |x - y| is the total variation distance (half L1 distance).

    Args:
        x (np.ndarray): 1D pmf of the first distribution with the same support
        y (np.ndarray): 1D pmf of the second distribution with the same support
        sigma (float, optional): standard deviation. Defaults to 1.0.

    Returns:
        float: Gaussian kernel value
    """

    # Convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x, y = process_tensor(x, y)

    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


def kernel_parallel_unpacked(
    x: np.ndarray,
    samples2: Iterator[np.ndarray],
    kernel: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Calculate the sum of the kernel values between x and all the samples in samples2

    Args:
        x (np.ndarray): "true sample"
        samples2 (Iterator[np.ndarray]): samples from the generator
        kernel (Callable[[np.ndarray, np.ndarray], float]): kernel function

    Returns:
        float: sum of kernel values
    """
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(
    t: Tuple[
        np.ndarray, Iterator[np.ndarray], Callable[[np.ndarray, np.ndarray], float]
    ]
) -> float:
    """Wrapper for kernel_parallel_unpacked

    Args:
        t (Tuple[np.ndarray, Iterator[np.ndarray], Callable[[np.ndarray, np.ndarray], float]]): tuple of arguments

    Returns:
        float: sum of kernel values
    """
    return kernel_parallel_unpacked(*t)


def disc(
    samples1: Iterator[np.ndarray],
    samples2: Iterator[np.ndarray],
    kernel: Callable[[np.ndarray, np.ndarray], float],
    is_parallel: bool = True,
    *args,
    **kwargs
) -> float:
    """Calculate the discrepancy between 2 samples

    Args:
        samples1 (Iterator[np.ndarray]): samples 1
        samples2 (Iterator[np.ndarray]): samples 2
        kernel (Callable[[np.ndarray, np.ndarray], float]): kernel function
        is_parallel (bool, optional): whether or not we use parallel processing. Defaults to True.

    Returns:
        float: discrepancy
    """
    d = 0
    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:  # parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for dist in executor.map(
                kernel_parallel_worker,
                [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1],
            ):
                d += dist
    d /= len(samples1) * len(samples2)  # normalize
    return d


def compute_mmd(
    samples1: Iterator[np.ndarray],
    samples2: Iterator[np.ndarray],
    kernel: Callable[[np.ndarray, np.ndarray], float],
    is_hist: bool = True,
    *args,
    **kwargs
) -> float:
    """Calculate the MMD (Maximum Mean Discrepancy) between two samples

    Args:
        samples1 (Iterator[np.ndarray]): samples 1
        samples2 (Iterator[np.ndarray]): samples 2
        kernel (Callable[[np.ndarray, np.ndarray], float]): kernel function
        is_hist (bool, optional): whether or not we normalize the input to transform
            it into histograms. Defaults to True.

    Returns:
        float: MMD
    """
    if is_hist:  # normalize histograms into pmf
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]
    return (
        disc(samples1, samples1, kernel, *args, **kwargs)
        + disc(samples2, samples2, kernel, *args, **kwargs)
        - 2 * disc(samples1, samples2, kernel, *args, **kwargs)
    )


def compute_emd(
    samples1: Iterator[np.ndarray],
    samples2: Iterator[np.ndarray],
    kernel: Callable[[np.ndarray, np.ndarray], float],
    is_hist: bool = True,
    *args,
    **kwargs
) -> Tuple[float, List[np.ndarray]]:
    """Calculate the EMD (Earth Mover Distance) between the average of two samples

    Args:
        samples1 (Iterator[np.ndarray]): samples 1
        samples2 (Iterator[np.ndarray]): samples 2
        kernel (Callable[[np.ndarray, np.ndarray], float]): kernel function
        is_hist (bool, optional): whether or not we normalize the input to transform
            it into histograms. Defaults to True.

    Returns:
        Tuple[float, List[np.ndarray]]: EMD and the average of the two samples
    """

    if is_hist:  # normalize histograms into pmf, take the average
        samples1 = [np.mean(samples1)]
        samples2 = [np.mean(samples2)]
    return disc(samples1, samples2, kernel, *args, **kwargs), [samples1[0], samples2[0]]


def preprocess(X: np.ndarray, max_len: int, is_hist: bool) -> np.ndarray:
    """Preprocess function for the kernel_compute function below

    Args:
        X (np.ndarray): input array
        max_len (int): max row length of the new array
        is_hist (bool): if the input array is an histogram

    Returns:
        np.ndarray: preprocessed output array
    """
    X_p = np.zeros((len(X), max_len))
    for i in range(len(X)):
        X_p[i, : len(X[i])] = X[i]

    if is_hist:
        row_sum = np.sum(X_p, axis=1)
        X_p = X_p / row_sum[:, None]

    return X_p


def kernel_compute(
    X: List[nx.Graph],
    Y: Optional[List[nx.Graph]] = None,
    is_hist: bool = True,
    metric: str = "linear",
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """Function to compute the kernel matrix with list of graphs as inputs and
    a custom metric

    Adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py

    Args:
        X (List[nx.Graph]): samples 1 (list of graphs)
        Y (Optional[List[nx.Graph]], optional): samples 2 (list of graphs). Defaults to None.
        is_hist (bool, optional): whether of not the input should be histograms (NOT IMPLEMENTED). Defaults to True.
        metric (str, optional): metric. Defaults to "linear".
        n_jobs (Optional[int], optional): number of jobs for parallel computing. Defaults to None.

    Returns:
        np.ndarray: kernel matrix
    """

    if metric == "nspdk":
        X = vectorize(X, complexity=4, discrete=True)

        if Y is not None:
            Y = vectorize(Y, complexity=4, discrete=True)

        return pairwise_kernels(X, Y, metric="linear", n_jobs=n_jobs)

    else:
        max_len = max([len(x) for x in X])
        if Y is not None:
            max_len = max(max_len, max([len(y) for y in Y]))

        X = preprocess(X, max_len, is_hist)

        if Y is not None:
            Y = preprocess(Y, max_len, is_hist)

        return pairwise_kernels(X, Y, metric=metric, n_jobs=n_jobs)


def compute_nspdk_mmd(
    samples1: List[nx.Graph],
    samples2: List[nx.Graph],
    metric: str,
    is_hist: bool = True,
    n_jobs: Optional[int] = None,
) -> float:
    """
    Compute the MMD between two samples of graphs using the NSPDK kernel

    Adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py

    Args:
        samples1 (List[nx.Graph]): samples 1 (list of graphs)
        samples2 (List[nx.Graph]): samples 2 (list of graphs)
        metric (str): metric
        is_hist (bool, optional): whether of not the input should be histograms (NOT IMPLEMENTED). Defaults to True.
        n_jobs (Optional[int], optional): number of jobs for parallel computing. Defaults to None.
    """

    X = kernel_compute(samples1, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Y = kernel_compute(samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Z = kernel_compute(
        samples1, Y=samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs
    )

    return np.average(X) + np.average(Y) - 2 * np.average(Z)


def process_tensor(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Process two tensors (vectors) to have the same size (support)

    Args:
        x (np.ndarray): vector 1
        y (np.ndarray): vector 2

    Returns:
        Tuple[np.ndarray, np.ndarray]: processed vectors
    """
    support_size = max(len(x), len(y))
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    return x, y

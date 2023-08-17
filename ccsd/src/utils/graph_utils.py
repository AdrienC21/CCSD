#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""graph_utils.py: utility functions for graph data (flag masking, quantization, etc.).

Adapted from Jo, J. & al (2022), almost left untouched.
"""

from typing import List, Optional, Union

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from rdkit import Chem

from ccsd.src.utils.errors import SymmetryError
from ccsd.src.utils.models_utils import get_ones
from ccsd.src.utils.mol_utils import bond_decoder


def mask_x(x: torch.Tensor, flags: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mask batch of node features with 0-1 flags tensor

    Args:
        x (torch.Tensor): batch of node features
        flags (Optional[torch.Tensor], optional): 0-1 flags tensor. Defaults to None.

    Returns:
        torch.Tensor: Mask batch of node features
    """
    if flags is None:
        flags = get_ones((x.shape[0], x.shape[1]), x.device)
    return x * flags[:, :, None]


def mask_adjs(adjs: torch.Tensor, flags: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mask batch of adjacency matrices with 0-1 flags tensor

    Args:
        adjs (torch.Tensor): batch of adjacency matrices.
            B x N x N or B x C x N x N
        flags (Optional[torch.Tensor], optional): 0-1 flags tensor. Defaults to None.
            B x N

    Returns:
        torch.Tensor: Mask batch of adjacency matrices
    """
    if flags is None:
        flags = get_ones((adjs.shape[0], adjs.shape[-1]), adjs.device)

    if len(adjs.shape) == 4:
        flags = flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs


def node_flags(adj: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Create flags tensor from graph dataset

    Args:
        adj (torch.Tensor): adjacency matrix
        eps (float, optional): threshold. Defaults to 1e-5.

    Returns:
        torch.Tensor: flags tensor
    """

    flags = torch.abs(adj).sum(-1).gt(eps).to(dtype=torch.float32)

    if len(flags.shape) == 3:
        flags = flags[:, 0, :]
    return flags


def init_features(init: str, adjs: torch.Tensor, nfeat: int = 10) -> torch.Tensor:
    """Create initial node features by initaliazing the adjacency matrix,
    creating a node flag matrix based on the initialization, and masking the
    node features with the node flag matrix

    Args:
        init (str): node feature initialization method
        adjs (torch.Tensor, optional): adjacency matrix.
        nfeat (int, optional): number of different features. Defaults to 10.

    Raises:
        ValueError: If number of features is larger than number of classes
        NotImplementedError: initialization method not implemented

    Returns:
        torch.Tensor: node features tensor
    """

    if init == "zeros":
        feature = torch.zeros(
            (adjs.size(0), adjs.size(1), nfeat), dtype=torch.float32, device=adjs.device
        )
    elif init == "ones":
        feature = get_ones((adjs.size(0), adjs.size(1), nfeat), adjs.device)
    elif init == "deg":
        feature = adjs.sum(dim=-1).to(torch.long)
        num_classes = nfeat
        try:
            feature = F.one_hot(feature, num_classes=num_classes).to(torch.float32)
        except:
            raise ValueError(
                f"Max number of feature ({feature.max().item()}) and number of classes ({num_classes}) missmatch"
            )
    else:
        raise NotImplementedError(
            f"{init} not implemented. Please select from [zeros, ones, deg]."
        )

    flags = node_flags(adjs)
    return mask_x(feature, flags)


def init_flags(
    graph_list: List[nx.Graph], config: EasyDict, batch_size: Optional[int] = None
) -> torch.Tensor:
    """Sample initial flags tensor from the training graph set

    Args:
        graph_list (List[nx.Graph]): list of graphs
        config (EasyDict): _description_
        batch_size (Optional[int], optional): batch size. Defaults to None.

    Returns:
        torch.Tensor: flag tensors
    """

    # Old code
    """
    if batch_size is None:  # get a default one from the config
        batch_size = config.data.batch_size
    max_node_num = config.data.max_node_num
    graph_tensor = graphs_to_tensor(graph_list, max_node_num)
    idx = np.random.randint(0, len(graph_list), batch_size)
    flags = node_flags(graph_tensor[idx])

    return flags
    """

    raise DeprecationWarning(
        "Use init_flags from the cc_utils instead. For graphs, let the parameter is_cc=False."
    )


def gen_noise(
    x: torch.Tensor, flags: Optional[torch.Tensor] = None, sym: bool = True
) -> torch.Tensor:
    """Generate noise

    Args:
        x (torch.Tensor): input tensor
        flags (Optional[torch.Tensor], optional): optional flags. Defaults to None.
        sym (bool, optional): symetric noise (for adjacency matrix). Defaults to True.

    Returns:
        torch.Tensor: generated noisy tensor
    """
    z = torch.randn_like(x)  # gaussian centered normal distribution
    if sym:
        z = z.triu(1)  # keep only upper triangular part
        z = z + z.transpose(-1, -2)
        z = mask_adjs(z, flags)
    else:
        z = mask_x(z, flags)
    return z


def quantize(t: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Quantize (clip) generated graphs regarding a threshold

    Args:
        t (torch.Tensor): original adjacency or rank2 incidence matrix
        thr (float, optional): threshold. Defaults to 0.5.

    Returns:
        torch.Tensor: quantized/cropped/clipped an adjacency or rank2 incidence matrix
    """
    t_ = torch.where(t < thr, torch.zeros_like(t), get_ones(t.shape, t.device))
    return t_


def quantize_mol(adjs: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Quantize generated molecules

    Args:
        adjs (Union[torch.Tensor, np.ndarray]): adjacency matrix
            adjs: 32 x 9 x 9

    Returns:
        np.ndarray: quantized array for molecules
    """
    if isinstance(adjs, torch.Tensor):
        adjs = adjs.detach().cpu()
    else:  # convert to tensor
        adjs = torch.tensor(adjs)
    adjs[adjs >= 2.5] = 3
    adjs[torch.bitwise_and(adjs >= 1.5, adjs < 2.5)] = 2
    adjs[torch.bitwise_and(adjs >= 0.5, adjs < 1.5)] = 1
    adjs[adjs < 0.5] = 0
    return np.array(adjs.to(torch.int64))


def adjs_to_graphs(
    adjs: Union[
        torch.Tensor,
        List[torch.Tensor],
        List[np.ndarray],
        List[List[List[Union[int, float]]]],
    ],
    is_cuda: bool = False,
) -> List[nx.Graph]:
    """Convert generated adjacency matrices to networkx graphs

    Args:
        adjs (Union[torch.Tensor, List[torch.Tensor], List[np.ndarray], List[List[List[Union[int, float]]]]]): Adjaency matrices
        is_cuda (bool, optional): are the tensor on CPU?. Defaults to False.

    Returns:
        List[nx.Graph]: list of graph representations
    """
    graph_list = []
    for adj in adjs:
        if is_cuda:
            adj = adj.detach().cpu().numpy()
        else:
            if isinstance(adj, torch.Tensor):
                adj = adj.detach().numpy()
            elif isinstance(adj, np.ndarray):
                pass
            elif isinstance(adj, list):
                adj = np.array(adj, dtype=np.float32)
        G = nx.from_numpy_array(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 1:
            G.add_node(1)
        graph_list.append(G)
    return graph_list


def check_sym(
    adjs: torch.Tensor, print_val: bool = False, epsilon: float = 1e-2
) -> None:
    """Check if the adjacency matrices are symmetric

    Args:
        adjs (torch.Tensor): adjacency matrices
        print_val (bool, optional): whether or not we print the symmetry error. Defaults to False.
        epsilon (float, optional): theshold for the sum of the absolute errors. Defaults to 1e-2.

    Raises:
        SymmetryError: If the sum of the absolute errors is greater than epsilon
    """
    sym_error = (adjs - adjs.transpose(-1, -2)).abs().sum([0, 1, 2])
    if not (sym_error < epsilon):
        raise SymmetryError(f"Tensor not symmetric: {sym_error:.4e}")
    if print_val:
        print(f"{sym_error:.4e}")


def pow_tensor(x: torch.Tensor, cnum: int) -> torch.Tensor:
    """Create higher order adjacency matrices

    Args:
        x (torch.Tensor): input tensor of shape B x N x N
        cnum (int): number of higher order matrices to create (made with powers of x)

    Returns:
        torch.Tensor: output higher order matrices of shape B x cnum x N x N
    """
    #
    x_ = x.clone()
    xc = [x.unsqueeze(1)]
    for _ in range(cnum - 1):
        x_ = torch.bmm(x_, x)
        xc.append(x_.unsqueeze(1))
    xc = torch.cat(xc, dim=1)

    return xc


def pad_adjs(ori_adj: np.ndarray, node_number: int) -> np.ndarray:
    """Create padded adjacency matrices

    Args:
        ori_adj (np.ndarray): original adjacency matrix
        node_number (int): number of desired nodes

    Raises:
        ValueError: if the original adjacency matrix is larger than the desired number of nodes (we can't pad)

    Returns:
        np.ndarray: Padded adjacency matrix
    """
    if not (ori_adj.size):  # empty
        return np.zeros((node_number, node_number), dtype=np.float32)
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:  # same shape
        return a
    if ori_len > node_number:
        raise ValueError(
            f"Original number of nodes {ori_len} is greater (>) that the desired number of nodes after padding {node_number}"
        )
    # Pad
    a = np.concatenate([a, np.zeros([ori_len, node_number - ori_len])], axis=-1)
    a = np.concatenate([a, np.zeros([node_number - ori_len, node_number])], axis=0)
    return a


def graphs_to_tensor(graph_list: List[nx.Graph], max_node_num: int) -> torch.Tensor:
    """Convert a list of graphs to a tensor

    Args:
        graph_list (List[nx.Graph]): List of graphs to convert to adjacency matrices tensors
        max_node_num (int): max number of nodes in all the graphs

    Returns:
        torch.Tensor: Tensor of adjacency matrices
    """
    adjs_list = []
    max_node_num = max_node_num  # memory issue

    for g in graph_list:
        assert isinstance(g, nx.Graph)
        node_list = []
        for v, feature in g.nodes.data("feature"):
            node_list.append(v)

        # convert to adj matrix
        adj = nx.to_numpy_array(g, nodelist=node_list)
        padded_adj = pad_adjs(adj, node_number=max_node_num)  # pad to max node number
        adjs_list.append(padded_adj)

    del graph_list

    adjs_np = np.asarray(adjs_list)  # concatenate the arrays
    del adjs_list

    adjs_tensor = torch.tensor(adjs_np, dtype=torch.float32)  # convert to tensor
    del adjs_np

    return adjs_tensor


def graphs_to_adj(graph: nx.Graph, max_node_num: int) -> torch.Tensor:
    """Convert a graph to an adjacency matrix

    Args:
        graph (nx.Graph): graph to convert to an adjacency matrix tensor
        max_node_num (int): maximum number of nodes

    Returns:
        torch.Tensor: Adjacency matrix as a tensor
    """
    max_node_num = max_node_num  # memory issue

    assert isinstance(graph, nx.Graph)
    node_list = []
    for v, feature in graph.nodes.data("feature"):
        node_list.append(v)

    adj = nx.to_numpy_array(graph, nodelist=node_list)
    padded_adj = pad_adjs(adj, node_number=max_node_num)

    adj = torch.tensor(padded_adj, dtype=torch.float32)
    del padded_adj

    return adj


def node_feature_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Convert a node feature matrix to a node pair feature matrix.
    Squared matrices where coeff i, j: concatenation of coeff i and coeff j of the associated
    node feature matrix

    Args:
        x (torch.Tensor): B x N x F  (F feature space)

    Returns:
        torch.Tensor: converted node feature matrix to node pair feature matrix with shape B x N x N x 2F
    """
    x_b = x.unsqueeze(-2).expand(x.size(0), x.size(1), x.size(1), -1)  # B x N x N x F
    x_pair = torch.cat([x_b, x_b.transpose(1, 2)], dim=-1)  # B x N x N x 2F

    return x_pair


def nxs_to_mols(graphs: List[nx.Graph]) -> List[Chem.Mol]:
    """Convert a list of nx graphs to a list of rdkit molecules

    Args:
        graphs (List[nx.Graph]): list of nx graphs

    Returns:
        List[Chem.Mol]: list of rdkit molecules
    """
    mols = []
    for g in graphs:
        mol = Chem.RWMol()
        for node, symbol in g.nodes.data("label"):
            mol.AddAtom(Chem.Atom(symbol))
        for atom_a, atom_b, bond_type in g.edges.data("label"):
            mol.AddBond(atom_a, atom_b, bond_decoder[bond_type])
        mols.append(mol)
    return mols

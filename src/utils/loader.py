#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""loader.py: code for loading the model, the optimizer, the scheduler, the loss function, etc
"""

from typing import Tuple, Union, List, Callable, Optional, Dict, Any

import torch
import random
import numpy as np
import networkx as nx
from easydict import EasyDict
from torch.utils.data import DataLoader
from toponetx.classes.combinatorial_complex import CombinatorialComplex

from src.models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_GMH
from src.models.ScoreNetwork_A import ScoreNetworkA
from src.models.ScoreNetwork_F import ScoreNetworkF
from src.sde import VPSDE, VESDE, subVPSDE, SDE
from src.losses import get_sde_loss_fn, get_sde_loss_fn_cc
from src.solver import get_pc_sampler, S4_solver
from src.evaluation.mmd import gaussian, gaussian_emd, gaussian_tv
from src.utils.ema import ExponentialMovingAverage
from src.utils.data_loader import dataloader, dataloader_cc
from src.utils.data_loader_mol import dataloader_mol, dataloader_mol_cc
from src.utils.cc_utils import get_rank2_dim


def load_seed(seed: int) -> int:
    """Apply the random seed to all libraries (torch, numpy, random)
    and make sure that the results are reproducible.

    Args:
        seed (int): seed to use

    Returns:
        int: return the seed
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def load_device() -> str:
    """Check if cuda is available and then return the device to use

    Returns:
        str: device to use
    """
    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))
    else:
        device = "cpu"
    return device


def load_model(params: Dict[str, Any]) -> torch.nn.Module:
    """Load the Score Network model from the parameters

    Args:
        params (dict): parameters to use

    Raises:
        ValueError: raise an error if the model is unknown

    Returns:
        torch.nn.Module: Score Network model to use
    """
    params_ = params.copy()
    model_type = params_.pop("model_type", None)
    if model_type == "ScoreNetworkX":
        model = ScoreNetworkX(**params_)
    elif model_type == "ScoreNetworkX_GMH":
        model = ScoreNetworkX_GMH(**params_)
    elif model_type == "ScoreNetworkA":
        model = ScoreNetworkA(**params_)
    elif model_type == "ScoreNetworkF":
        model = ScoreNetworkF(**params_)
    else:
        raise ValueError(
            f"Model Name <{model_type}> is unknown. Please select from [ScoreNetworkX, ScoreNetworkX_GMH, ScoreNetworkA, ScoreNetworkF]"
        )
    return model


def load_model_optimizer(
    params: Dict[str, Any], config_train: EasyDict, device: Union[str, List[str]]
) -> Tuple[
    Union[torch.nn.Module, torch.nn.DataParallel],
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    """Return the model, the optimizer and the scheduler in function of the parameters

    Args:
        params (Dict[str, Any]): model parameters
        config_train (EasyDict): configuration for training
        device (str): device to use

    Returns:
        Tuple[Union[torch.nn.Module, torch.nn.DataParallel], torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]: return the model, the optimizer and the scheduler
    """
    model = load_model(params)
    if isinstance(device, list):  # check for multi-gpu
        if len(device) > 1:  # multi-gpu
            model = torch.nn.DataParallel(model, device_ids=device)
        model = model.to(f"cuda:{device[0]}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config_train.lr, weight_decay=config_train.weight_decay
    )
    scheduler = None
    if config_train.lr_schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config_train.lr_decay
        )

    return model, optimizer, scheduler


def load_ema(model: torch.nn.Module, decay: float = 0.999) -> ExponentialMovingAverage:
    """Create an exponential moving average object for the model's parameters

    Args:
        model (torch.nn.Module): model used to train the model
        decay (float, optional): decay parameter. Defaults to 0.999.

    Returns:
        ExponentialMovingAverage: exponential moving average object for the model's parameters
    """
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema


def load_ema_from_ckpt(
    model: torch.nn.Module, ema_state_dict: Dict[str, Any], decay: float = 0.999
) -> ExponentialMovingAverage:
    """Load the exponential moving average object for the model's parameters from a checkpoint

    Args:
        model (torch.nn.Module): model used to train the model
        ema_state_dict (Dict[str, Any]): parameters of the exponential moving average
        decay (float, optional): decay parameter. Defaults to 0.999.

    Returns:
        ExponentialMovingAverage: exponential moving average object for the model's parameters
    """
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    ema.load_state_dict(ema_state_dict)
    return ema


def load_data(
    config: EasyDict,
    get_list: bool = False,
    is_cc: bool = False,
) -> Union[
    Tuple[DataLoader, DataLoader],
    Union[
        Tuple[List[nx.Graph], List[nx.Graph]],
        Tuple[List[CombinatorialComplex], List[CombinatorialComplex]],
    ],
]:
    """Return a DataLoader object for training based on the configuration

    Args:
        config (EasyDict): configuration for training
        get_list (bool, optional): if True, returns lists of graph or combinatorial complexes instead of dataloaders. Defaults to False.
        is_cc (bool, optional): if True, the dataset is made of combinatorial complexes. Defaults to False.

    Returns:
        Union[Tuple[DataLoader, DataLoader], Union[Tuple[List[nx.Graph], List[nx.Graph]], Tuple[List[CombinatorialComplex], List[CombinatorialComplex]]]]: DataLoader object or list of objects for training
    """
    if config.data.data in ["QM9"]:
        if not (is_cc):
            return dataloader_mol(config, get_list)
        return dataloader_mol_cc(config, get_list)
    else:
        if not (is_cc):
            return dataloader(config, get_list)
        return dataloader_cc(config, get_list)


def load_batch(
    batch: List[torch.Tensor], device: Union[str, List[str]], is_cc: bool = False
) -> Union[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """Load the batch on the device

    Args:
        batch (List[torch.Tensor]): input batch
        device (Union[str, List[str]]): device to use
        is_cc (bool, optional): if True, the elements of the input batch are combinatorial complexes. Defaults to False.

    Returns:
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: input batch on the device
    """
    device_id = f"cuda:{device[0]}" if isinstance(device, list) else device
    x_b = batch[0].to(device_id)
    adj_b = batch[1].to(device_id)
    if not (is_cc):
        return x_b, adj_b
    rank2_b = batch[2].to(device_id)
    return x_b, adj_b, rank2_b


def load_sde(config_sde: EasyDict) -> SDE:
    """Load the stochastic differential equation (SDE) from the configuration

    Args:
        config_sde (EasyDict): configuration for the SDE

    Raises:
        NotImplementedError: raise an error if the SDE is unknown

    Returns:
        SDE: SDE to use
    """
    sde_type = config_sde.type
    beta_min = config_sde.beta_min
    beta_max = config_sde.beta_max
    num_scales = config_sde.num_scales

    if sde_type == "VP":
        sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    elif sde_type == "VE":
        sde = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_scales)
    elif sde_type == "subVP":
        sde = subVPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    else:
        raise NotImplementedError(f"SDE class {sde_type} not (yet) supported.")
    return sde


def load_loss_fn(
    config: EasyDict,
    is_cc: bool = False,
) -> Union[
    Callable[
        [
            torch.nn.Module,
            torch.nn.Module,
            torch.Tensor,
            torch.Tensor,
        ],
        Tuple[torch.Tensor, torch.Tensor],
    ],
    Callable[
        [
            torch.nn.Module,
            torch.nn.Module,
            torch.nn.Module,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
]:
    """Load the loss function from the configuration

    Args:
        config (EasyDict): configuration to use
        is_cc (bool, optional): if True, loss function for combinatorial complexes. Defaults to False.

    Returns:
        Union[Callable[[torch.nn.Module, torch.nn.Module, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], Callable[[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]: loss function that returns 2 or 3 losses, for x, adj and rank2 if cc
    """
    reduce_mean = config.train.reduce_mean
    sde_x = load_sde(config.sde.x)
    sde_adj = load_sde(config.sde.adj)

    if not (is_cc):
        loss_fn = get_sde_loss_fn(
            sde_x,
            sde_adj,
            train=True,
            reduce_mean=reduce_mean,
            continuous=True,
            likelihood_weighting=False,
            eps=config.train.eps,
        )
    else:
        sde_rank2 = load_sde(config.sde.rank2)
        d_min = config.data.d_min
        d_max = config.data.d_max
        loss_fn = get_sde_loss_fn_cc(
            sde_x,
            sde_adj,
            sde_rank2,
            d_min=d_min,
            d_max=d_max,
            train=True,
            reduce_mean=reduce_mean,
            continuous=True,
            likelihood_weighting=False,
            eps=config.train.eps,
        )
    return loss_fn


def load_sampling_fn(
    config_train: EasyDict,
    config_module: EasyDict,
    config_sample: EasyDict,
    device: Union[str, List[str]],
    is_cc: bool = False,
    d_min: Optional[int] = None,
    d_max: Optional[int] = None,
) -> Union[
    Callable[
        [torch.nn.Module, torch.nn.Module, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, float],
    ],
    Callable[
        [torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float],
    ],
]:
    """Load the sampling function from the configuration

    Args:
        config_train (EasyDict): configuration for training
        config_module (EasyDict): configuration for the module
        config_sample (EasyDict): configuration for the sampling
        device (Union[str, List[str]]): device to use
        is_cc (bool, optional): if True, we sample combinatorial complexes. Defaults to False.
        d_min (Optional[int], optional): minimum size of rank2 cells (for cc). Defaults to None.
        d_max (Optional[int], optional): maximum size of rank2 cells (for cc). Defaults to None.

    Returns:
        Union[Callable[[torch.nn.Module, torch.nn.Module, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, float]], Callable[[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]]: sampling function
    """

    sde_x = load_sde(config_train.sde.x)
    sde_adj = load_sde(config_train.sde.adj)
    if is_cc:
        sde_rank2 = load_sde(config_train.sde.rank2)
    max_node_num = config_train.data.max_node_num

    device_id = f"cuda:{device[0]}" if isinstance(device, list) else device

    # Get sampler
    if config_module.predictor == "S4":
        get_sampler = S4_solver
    else:
        get_sampler = get_pc_sampler

    # Get shape in function of dataset
    if config_train.data.data in ["QM9"]:
        shape_x = (10000, max_node_num, config_train.data.max_feat_num)
        shape_adj = (10000, max_node_num, max_node_num)
        if is_cc:
            rank2_dim = get_rank2_dim(max_node_num, d_min, d_max)
            shape_rank2 = (10000, rank2_dim[0], rank2_dim[1])
    else:
        shape_x = (
            config_train.data.batch_size,
            max_node_num,
            config_train.data.max_feat_num,
        )
        shape_adj = (config_train.data.batch_size, max_node_num, max_node_num)
        if is_cc:
            rank2_dim = get_rank2_dim(max_node_num, d_min, d_max)
            shape_rank2 = (config_train.data.batch_size, rank2_dim[0], rank2_dim[1])

    # Get sampling function
    if not (is_cc):
        sampling_fn = get_sampler(
            sde_x=sde_x,
            sde_adj=sde_adj,
            shape_x=shape_x,
            shape_adj=shape_adj,
            predictor=config_module.predictor,
            corrector=config_module.corrector,
            snr=config_module.snr,
            scale_eps=config_module.scale_eps,
            n_steps=config_module.n_steps,
            probability_flow=config_sample.probability_flow,
            continuous=True,
            denoise=config_sample.noise_removal,
            eps=config_sample.eps,
            device=device_id,
        )
    else:
        sampling_fn = get_sampler(
            sde_x=sde_x,
            sde_adj=sde_adj,
            shape_x=shape_x,
            shape_adj=shape_adj,
            predictor=config_module.predictor,
            corrector=config_module.corrector,
            snr=config_module.snr,
            scale_eps=config_module.scale_eps,
            n_steps=config_module.n_steps,
            probability_flow=config_sample.probability_flow,
            continuous=True,
            denoise=config_sample.noise_removal,
            eps=config_sample.eps,
            device=device_id,
            is_cc=is_cc,
            sde_rank2=sde_rank2,
            shape_rank2=shape_rank2,
            d_min=d_min,
            d_max=d_max,
        )
    return sampling_fn


def load_model_params(
    config: EasyDict,
    is_cc: bool = False,
) -> Union[
    Tuple[Dict[str, Any], Dict[str, Any]],
    Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]],
]:
    """Load the model parameters from the configuration

    Args:
        config (EasyDict): configuration to use
        is_cc (bool, optional): whether to model using combinatorial complexes. Defaults to False.

    Returns:
        Union[Tuple[Dict[str, Any], Dict[str, Any]], Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]: parameters for x, adj, and rank-2 cells if cc
    """

    assert is_cc == config.is_cc, "is_cc should be the same in config and function call"

    config_m = config.model
    max_feat_num = config.data.max_feat_num
    max_node_num = config.data.max_node_num

    if "GMH" in config_m.x:
        params_x = {
            "is_cc": is_cc,
            "model_type": config_m.x,
            "max_feat_num": max_feat_num,
            "depth": config_m.depth,
            "nhid": config_m.nhid,
            "num_linears": config_m.num_linears,
            "c_init": config_m.c_init,
            "c_hid": config_m.c_hid,
            "c_final": config_m.c_final,
            "adim": config_m.adim,
            "num_heads": config_m.num_heads,
            "conv": config_m.conv,
            "use_bn": config_m.use_bn,
        }
    else:
        params_x = {
            "is_cc": is_cc,
            "model_type": config_m.x,
            "max_feat_num": max_feat_num,
            "depth": config_m.depth,
            "nhid": config_m.nhid,
            "use_bn": config_m.use_bn,
        }
    params_adj = {
        "is_cc": is_cc,
        "model_type": config_m.adj,
        "max_feat_num": max_feat_num,
        "max_node_num": config.data.max_node_num,
        "nhid": config_m.nhid,
        "num_layers": config_m.num_layers,
        "num_linears": config_m.num_linears,
        "c_init": config_m.c_init,
        "c_hid": config_m.c_hid,
        "c_final": config_m.c_final,
        "adim": config_m.adim,
        "num_heads": config_m.num_heads,
        "conv": config_m.conv,
        "use_bn": config_m.use_bn,
    }
    if not (is_cc):
        return params_x, params_adj
    # If is_cc, also load rank-2 parameters
    d_min = config.data.d_min
    d_max = config.data.d_max
    params_rank2 = {
        "is_cc": config.is_cc,
        "model_type": config_m.rank2,
        "num_layers_mlp": config_m.num_layers_mlp,
        "num_layers": config_m.num_layers,
        "num_linears": config_m.num_linears,
        "nhid": config_m.nhid,
        "c_hid": config_m.c_hid,
        "c_final": config_m.c_final,
        "cnum": config_m.cnum,
        "max_node_num": max_node_num,
        "d_min": d_min,
        "d_max": d_max,
        "use_hodge_mask": config_m.use_hodge_mask,
        "use_bn": config_m.use_bn,
    }
    return params_x, params_adj, params_rank2


def load_ckpt(
    config: EasyDict,
    device: Union[str, List[str]],
    ts: Optional[str] = None,
    return_ckpt: bool = False,
    is_cc: bool = False,
) -> Dict[str, Any]:
    """Load the checkpoint from the configuration

    Args:
        config (EasyDict): configuration to use
        device (Union[str, List[str]]): device to use
        ts (Optional[str], optional): timestamp (checkpoint name). Defaults to None.
        return_ckpt (bool, optional): if True, add the checkpoint in the resulting dictionary (key: "ckpt"). Defaults to False.
        is_cc (bool, optional): whether to model using combinatorial complexes. Defaults to False.

    Returns:
        Dict[str, Any]: _description_
    """
    device_id = f"cuda:{device[0]}" if isinstance(device, list) else device
    ckpt_dict = {}
    if ts is not None:
        config.ckpt = ts
    path = f"./checkpoints/{config.data.data}/{config.ckpt}.pth"
    ckpt = torch.load(path, map_location=device_id)
    print(f"{path} loaded")
    ckpt_dict = {
        "config": ckpt["model_config"],
        "params_x": ckpt["params_x"],
        "x_state_dict": ckpt["x_state_dict"],
        "params_adj": ckpt["params_adj"],
        "adj_state_dict": ckpt["adj_state_dict"],
    }
    if is_cc:
        ckpt_dict["params_rank2"] = ckpt["params_rank2"]
        ckpt_dict["rank2_state_dict"] = ckpt["rank2_state_dict"]
    if config.sample.use_ema:
        ckpt_dict["ema_x"] = ckpt["ema_x"]
        ckpt_dict["ema_adj"] = ckpt["ema_adj"]
        if is_cc:
            ckpt_dict["ema_rank2"] = ckpt["ema_rank2"]
    if return_ckpt:
        ckpt_dict["ckpt"] = ckpt
    return ckpt_dict


def load_model_from_ckpt(
    params: Dict[str, Any],
    state_dict: Dict[str, Any],
    device: Union[str, List[str], List[int]],
) -> Union[torch.nn.Module, torch.nn.DataParallel]:
    """Load the model from the checkpoint

    Args:
        params (Dict[str, Any]): parameters of the model
        state_dict (Dict[str, Any]): state dictionary of the model
        device (Union[str, List[str], List[int]]): device to use

    Returns:
        Union[torch.nn.Module, torch.nn.DataParallel]: loaded model
    """
    model = load_model(params)
    if "module." in list(state_dict.keys())[0]:
        # strip 'module.' at front; for DataParallel models
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if isinstance(device, list):
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device)
        new_device = f"cuda:{device[0]}" if "cuda" not in device[0] else device[0]
        model = model.to(new_device)
    else:
        model = model.to(device)
    return model


def load_eval_settings(
    data: Any, orbit_on: bool = True
) -> Tuple[List[str], Dict[str, Callable[[np.ndarray, np.ndarray], float]]]:
    """Load the evaluation settings from the configuration

    Args:
        data (Any): dataset to use. UNUSED HERE.
        orbit_on (bool, optional): whether to use orbit distance. UNUSED HERE. Defaults to True.

    Returns:
        Tuple[List[str], Dict[str, Callable[[np.ndarray, np.ndarray], float]]]: methods and kernels, used for generic graph generation
    """

    # Settings for generic graph generation

    # Methods to use (from [degree, cluster, orbit, spectral, nspdk], see evaluation/stats.py)
    methods = ["degree", "cluster", "orbit", "spectral"]
    # Kernels to use for each method (from [gaussian, gaussian_emd, gaussian_tv], see evaluation/mmd.py)
    kernels = {
        "degree": gaussian_emd,
        "cluster": gaussian_emd,
        "orbit": gaussian,
        "spectral": gaussian_emd,
    }
    return methods, kernels

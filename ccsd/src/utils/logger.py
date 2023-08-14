#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""logger.py: utility functions for logging.

Adapted from Jo, J. & al (2022), almost left untouched.
"""

import os
from typing import Any, List, Optional, Tuple, Union

import torch
from easydict import EasyDict


class Logger:
    """Logger class for logging to a file."""

    def __init__(self, filepath: str, mode: str, lock: Optional[Any] = None) -> None:
        """Initialize the Logger class.

        Args:
            filepath (str): the file where to write
            mode (str): can be 'w' or 'a'
            lock (Optional[Any], optional): pass a shared lock for multi process write access. Defaults to None.
        """
        self.filepath = filepath
        if mode not in ("w", "a"):
            assert False, "Mode must be one of w, r or a"
        else:
            self.mode = mode
        self.lock = lock

    def __repr__(self) -> str:
        """Return the string representation of the Logger class.

        Returns:
            str: the string representation of the Logger class
        """
        return f"Logger(filepath={self.filepath}, mode={self.mode})"

    def log(self, str: str, verbose: bool = True) -> None:
        """Log a string to the file and optionally print it

        Args:
            str (str): string to log
            verbose (bool, optional): whether or not we print the message. Defaults to True.
        """
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + "\n")

        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()

        if verbose:
            print(str)


def set_log(
    config: EasyDict, is_train: bool = True, folder: str = "./"
) -> Tuple[str, str, str]:
    """Set the log folder name, log directory and checkpoint directory

    Args:
        config (EasyDict): the config object
        is_train (bool, optional): True if we are training, False if we are sampling. Defaults to True.
        folder (str, optional): the general saving folder. Defaults to "./".

    Returns:
        Tuple[str, str, str]: the name of the folder, the log directory and the checkpoint directory of the log
    """

    data = config.data.data
    exp_name = config.train.name

    log_folder_name = os.path.join(*[data, exp_name])
    root = "logs_train" if is_train else "logs_sample"
    log_dir = os.path.join(folder, f"{root}", f"{log_folder_name}")
    if not (os.path.isdir(log_dir)):
        os.makedirs(log_dir)

    ckpt_dir = os.path.join(folder, "checkpoints", f"{data}")
    if not (os.path.isdir(ckpt_dir)) and is_train:
        os.makedirs(ckpt_dir)

    print(100 * "-")
    print("Make Directory {} in Logs".format(log_folder_name))

    return log_folder_name, log_dir, ckpt_dir


def check_log(log_folder_name: str, log_name: str) -> bool:
    """Check if a log file exists

    Args:
        log_folder_name (str): given log folder name
        log_name (str): given log name

    Returns:
        bool: True if the log file exists, False otherwise
    """
    filepath = os.path.join(*["logs_sample", log_folder_name, f"{log_name}.log"])
    return os.path.isfile(filepath)


def data_log(logger: Logger, config: EasyDict) -> None:
    """Log the current configuration

    Args:
        logger (Logger): Logger object
        config (EasyDict): current configuration used
    """
    logger.log(
        f"[{config.data.data}]   init={config.data.init} ({config.data.max_feat_num})   seed={config.seed}   batch_size={config.data.batch_size}"
    )


def sde_log(logger: Logger, config_sde: EasyDict, is_cc: bool = False) -> None:
    """Log the current SDE configuration

    Args:
        logger (Logger): Logger object
        config_sde (EasyDict): sde configuration
        is_cc (bool, optional): True if we are modelling with combinatorial complexes. Defaults to False.
    """
    sde_x = config_sde.x
    sde_adj = config_sde.adj
    to_log = (
        f"(x:{sde_x.type})=({sde_x.beta_min:.2f}, {sde_x.beta_max:.2f}) N={sde_x.num_scales} "
        f"(adj:{sde_adj.type})=({sde_adj.beta_min:.2f}, {sde_adj.beta_max:.2f}) N={sde_adj.num_scales} "
    )
    if is_cc:
        sde_rank2 = config_sde.rank2
        to_log += f"(rank2:{sde_rank2.type})=({sde_rank2.beta_min:.2f}, {sde_rank2.beta_max:.2f}) N={sde_rank2.num_scales}"
    logger.log(to_log)


def model_log(logger: Logger, config: EasyDict, is_cc: bool = False) -> None:
    """Log the current model configuration

    Args:
        logger (Logger): Logger object
        config (EasyDict): current configuration used
        is_cc (bool, optional): True if we are modelling with combinatorial complexes. Defaults to False.
    """
    config_m = config.model
    line1 = f"({config_m.x})+({config_m.adj}={config_m.conv},{config_m.num_heads})"
    if is_cc:
        h_mask = "hodge mask" if config_m.hodge_mask else "no hodge mask"
        line1 += (
            f"+({config_m.rank2}={h_mask}, {config_m.num_layers_mlp} {config_m.cnum})"
        )
    line1 += "   : "
    model_log = (
        line1
        + f"depth={config_m.depth} adim={config_m.adim} nhid={config_m.nhid} layers={config_m.num_layers} "
        + f"linears={config_m.num_linears} c=({config_m.c_init} {config_m.c_hid} {config_m.c_final})"
    )
    logger.log(model_log)


def device_log(
    logger: Logger, device: Union[str, List[int], List[str], List[torch.device]]
) -> None:
    """Log the device(s) that will be used as detected by PyTorch

    Args:
        logger (Logger): Logger object
        device (Union[str, List[int], List[str], List[torch.device]]): device(s) used as detected
    """
    print(100 * "-")
    logger.log(f"Using device: {device}")


def start_log(logger: Logger, config: EasyDict) -> None:
    """Log initial message with the configuration

    Args:
        logger (Logger): Logger object
        config (EasyDict): configuration used
    """
    logger.log(100 * "-")
    data_log(logger, config)
    logger.log(100 * "-")


def train_log(logger: Logger, config: EasyDict) -> None:
    """Log configuration used for training

    Args:
        logger (Logger): Logger object
        config (EasyDict): configuration used
    """
    logger.log(
        f"lr={config.train.lr} schedule={config.train.lr_schedule} ema={config.train.ema} "
        f"epochs={config.train.num_epochs} reduce={config.train.reduce_mean} eps={config.train.eps}"
    )
    model_log(logger, config)
    sde_log(logger, config.sde)
    logger.log(100 * "-")


def sample_log(logger: Logger, config: EasyDict) -> None:
    """Log configuration used for sampling

    Args:
        logger (Logger): Logger object
        config (EasyDict): configuration used
    """
    sample_log = (
        f"({config.sampler.predictor})+({config.sampler.corrector}): "
        f"eps={config.sample.eps} denoise={config.sample.noise_removal} "
        f"ema={config.sample.use_ema} "
    )
    if config.sampler.corrector == "Langevin":  # add Langevin's parameters
        sample_log += (
            f"|| snr={config.sampler.snr} seps={config.sampler.scale_eps} "
            f"n_steps={config.sampler.n_steps} "
        )
    logger.log(sample_log)
    logger.log(100 * "-")


def get_nb_parameters(model: torch.nn.Module) -> int:
    """Get the number of parameters of the model.

    Args:
        model (torch.nn.Module): model.

    Returns:
        int: number of parameters of the model.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_parameters_log(logger: Logger, models: List[torch.nn.Module]) -> None:
    """Print the number of parameters of the models and the total number of parameters.

    Args:
        logger (Logger): Logger object
        models (List[torch.nn.Module]): list of models.
    """

    model_parameters = [
        (model.__class__.__name__, get_nb_parameters(model)) for model in models
    ]
    total_parameters = sum(nb_param for _, nb_param in model_parameters)

    logger.log(100 * "-")
    logger.log("\nNumber of parameters:\n")
    for model_name, nb_param in model_parameters:
        logger.log(f"\t{model_name}: {nb_param}\n")
    logger.log(f"\nTotal: {total_parameters}\n")
    logger.log(100 * "-")

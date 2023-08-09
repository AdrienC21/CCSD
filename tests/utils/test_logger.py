#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_logger.py: test functions for logger.py
"""

import os

import pytest
import pathlib
from easydict import EasyDict
from pytest import CaptureFixture
from threading import Lock

from src.utils.logger import Logger, set_log


def test_logger_init() -> None:
    """Test the initialization of the Logger class."""
    filepath = "test.log"
    mode = "w"
    logger = Logger(filepath, mode)
    assert logger.filepath == filepath
    assert logger.mode == mode
    assert logger.lock is None


def test_logger_init_with_lock() -> None:
    """Test the initialization of the Logger class with a lock object."""
    filepath = "test.log"
    mode = "w"
    lock = Lock()
    logger = Logger(filepath, mode, lock)
    assert logger.filepath == filepath
    assert logger.mode == mode
    assert logger.lock == lock


def test_logger_invalid_mode() -> None:
    """Test the initialization of the Logger class with an invalid mode."""
    filepath = "test.log"
    invalid_mode = "invalid"
    with pytest.raises(AssertionError):
        logger = Logger(filepath, invalid_mode)


def test_log_method(tmpdir: pathlib.Path) -> None:
    """Test the log method.

    Args:
        tmpdir (pathlib.Path): temporary directory to write the log
    """
    # Create a temporary test file to write the log
    test_log_file = tmpdir.join("test_log.txt")

    # Initialize the logger
    logger = Logger(test_log_file, "w")

    # Test the log method with verbose=True
    test_string = "Test log message"
    logger.log(test_string, verbose=True)

    # Check if the log file is created and contains the log message
    assert test_log_file.read() == f"{test_string}\n"


def test_log_method_without_verbose(
    tmpdir: pathlib.Path, capsys: CaptureFixture[str]
) -> None:
    """Test the log method without verbose.

    Args:
        tmpdir (pathlib.Path): temporary directory to write the log
        capsys (CaptureFixture[str]): Capsys object to read the captured output
    """
    # Create a temporary test file to write the log
    test_log_file = tmpdir.join("test_log.txt")

    # Initialize the logger
    logger = Logger(test_log_file, "w")

    # Test the log method with verbose=False
    test_string = "Test log message"
    logger.log(test_string, verbose=False)

    # Check if the log file is created and contains the log message
    assert test_log_file.read() == f"{test_string}\n"

    # Check if the log message is not printed to the console (captured by capsys)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_log_method_with_lock(tmpdir: pathlib.Path) -> None:
    """Test the log method with a lock.

    Args:
        tmpdir (pathlib.Path): temporary directory to write the log
    """
    # Create a temporary test file to write the log
    test_log_file = tmpdir.join("test_log.txt")

    # Initialize the logger with a lock
    lock = Lock()
    logger = Logger(test_log_file, "w", lock)

    # Test the log method with verbose=True and lock
    test_string = "Test log message with lock"
    logger.log(test_string, verbose=True)

    # Check if the log file is created and contains the log message
    assert test_log_file.read() == f"{test_string}\n"


def test_log_method_exception(tmpdir: pathlib.Path) -> None:
    """Test the log method with an exception.

    Args:
        tmpdir (pathlib.Path): temporary directory to write the log

    Raises:
        Exception: Simulated exception
    """
    # Create a temporary test file to write the log
    test_log_file = tmpdir.join("test_log.txt")

    # Initialize the logger
    logger = Logger(test_log_file, "w")

    # Test the log method with an exception raised while writing to the file
    test_string = "Test log message with exception"
    with pytest.raises(Exception):
        logger.log(test_string, verbose=True)
        # Simulate an exception while writing to the file
        raise Exception("Simulated exception")

    # Check if the log file is created and contains the log message before the exception
    assert test_log_file.read() == f"{test_string}\n"


def test_set_log(tmpdir: pathlib.Path) -> None:
    """Test the set_log function.

    Args:
        tmpdir (pathlib.Path): temporary directory to write the log (UNUSED)
    """
    # Test set_log function
    data = "test_data"
    exp_name = "test_experiment"
    config = EasyDict({"data": {"data": data}, "train": {"name": exp_name}})
    is_train = True
    root = "logs_train" if is_train else "logs_sample"
    log_folder_name, log_dir, ckpt_dir = set_log(config, is_train=is_train)

    # Check if the log folder name, log directory, and checkpoint directory are correct
    assert log_folder_name == os.path.join(data, exp_name)
    assert os.path.isdir(log_dir)
    assert os.path.isdir(ckpt_dir)
    dir_to_delete = [
        os.path.join(f"./{root}/{data}/{exp_name}/"),
        os.path.join(f"./checkpoints/{data}"),
    ]
    for dir_del in dir_to_delete:
        if os.path.exists(dir_del):  # remove the folders created by set_log
            os.removedirs(dir_del)

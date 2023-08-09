#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_errors.py: test functions for errors.py
"""

import pytest

from src.utils.errors import SymmetryError


def test_symmetry_error() -> None:
    """Test the SymmetryError class.

    Raises:
        SymmetryError: if the error message is not the same as the one
    """
    error_message = "Matrix is not symmetric"
    with pytest.raises(SymmetryError) as exc_info:
        raise SymmetryError(error_message)

    assert str(exc_info.value) == error_message

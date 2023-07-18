#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""errors.py: contains custom exceptions.
"""

class SymmetryError(Exception):
    """Exception raised for when a matrix is not symmetric.

    Attributes:
        message -- more detailed explanation of the error
    """

    def __init__(self, message: str = "") -> None:
        """Raises a SymmetryError.

        Args:
            message (str, optional): more detailed explanation of the error. Defaults to "".
        """
        self.message = message
        super().__init__(self.message)

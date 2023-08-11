#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_time_utils.py: test functions for time_utils.py
"""

from datetime import datetime

import pytest
import pytz
from freezegun import freeze_time

from ccsd.src.utils.time_utils import get_time


@freeze_time("2023-07-28 14:35:30")
def test_get_time_valid_timezone() -> None:
    """Test get_time function with a valid timezone."""

    valid_timezone = "Europe/London"
    expected_time = "Jul28-15-35-30"  # London is UTC+1
    result = get_time(valid_timezone)

    # Assert that the result matches the expected time
    assert result == expected_time


def test_get_time_invalid_timezone() -> None:
    """Test get_time function with an invalid timezone."""

    invalid_timezone = "Invalid Timezone"

    # Call the function with the invalid timezone and expect a ValueError
    with pytest.raises(ValueError):
        get_time(invalid_timezone)

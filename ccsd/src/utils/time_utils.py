#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""time_utils.py: utility functions for time operations.
"""

import datetime

import pytz


def get_time(timezone: str = "Europe/London") -> str:
    if not (timezone in pytz.all_timezones):
        raise ValueError(
            "Invalid timezone {timezone}. You can get a list of all available timezones by typing: pytz.all_timezones."
        )
    tz = pytz.timezone(timezone)  # get timezone
    utc_time = datetime.datetime.utcnow()  # get the current time in UTC
    local_time = pytz.utc.localize(utc_time, is_dst=None).astimezone(tz)  # convert time
    # Format time (e.g. Jul28-14-27-30). This is used to name the log and checkpoint files.
    # No ':' in the time format because Windows does not allow ':' in filenames.
    ts = local_time.strftime("%b%d-%H-%M-%S")  # format time
    return ts

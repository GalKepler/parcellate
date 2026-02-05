"""Shared utility functions for interfaces.

This module provides shared utility functions for the interfaces.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable


def _parse_log_level(value: str | int | None) -> int:
    """Return a logging level from common string/int inputs.

    Parameters
    ----------
    value
        The value to parse.

    Returns
    -------
    int
        The logging level.

    Examples
    --------
    >>> _parse_log_level("INFO")
    20
    >>> _parse_log_level("DEBUG")
    10
    >>> _parse_log_level(logging.WARNING)
    30
    >>> _parse_log_level(None)
    20
    """
    if value is None:
        return logging.INFO
    if isinstance(value, int):
        return value
    return getattr(logging, str(value).upper(), logging.INFO)


def _as_list(value: Iterable[str] | str | None) -> list[str] | None:
    """Normalize configuration values into a list of strings.

    Parameters
    ----------
    value
        The value to normalize.

    Returns
    -------
    list[str] | None
        The normalized list of strings, or None if the input is None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)

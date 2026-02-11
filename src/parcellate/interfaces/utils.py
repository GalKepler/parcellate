"""Shared utility functions for interfaces.

This module provides shared utility functions for the interfaces.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from parcellate.interfaces.models import AtlasDefinition

logger = logging.getLogger(__name__)


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


def parse_atlases(
    atlas_configs: list[dict],
    default_space: str | None = None,
) -> list[AtlasDefinition]:
    """Parse atlas definitions from configuration.

    Parameters
    ----------
    atlas_configs
        List of atlas configuration dictionaries. Each dict should have
        'name' and 'path' keys, and optionally 'lut', 'space', and 'resolution'.
    default_space
        Default space to use if not specified in the config. If None, no
        default is applied.

    Returns
    -------
    list[AtlasDefinition]
        List of parsed atlas definitions.
    """
    atlases = []
    for cfg in atlas_configs:
        name = cfg.get("name")
        path = cfg.get("path")
        if not name or not path:
            logger.warning("Skipping atlas with missing name or path: %s", cfg)
            continue
        lut = cfg.get("lut")
        lut_path = Path(lut).expanduser().resolve() if lut else None
        space = cfg.get("space", default_space)
        resolution = cfg.get("resolution")

        atlases.append(
            AtlasDefinition(
                name=name,
                nifti_path=Path(path).expanduser().resolve(),
                lut=lut_path,
                space=space,
                resolution=resolution,
            )
        )
    return atlases

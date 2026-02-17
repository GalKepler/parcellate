"""Shared utility functions for interfaces.

This module provides shared utility functions for the interfaces.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from parcellate.interfaces.models import AtlasDefinition

logger = logging.getLogger(__name__)

_BUILTIN_MASK_NAMES: frozenset[str] = frozenset({"gm", "wm", "brain"})


def _mask_label(mask: Path | str | None) -> str | None:
    """Return the BIDS mask entity label for a given mask value.

    Parameters
    ----------
    mask
        Builtin mask name string (e.g. ``"gm"``), a filesystem :class:`~pathlib.Path`,
        or ``None``.

    Returns
    -------
    str | None
        * The builtin name (``"gm"``, ``"wm"``, ``"brain"``) if *mask* is one of those strings.
        * ``"custom"`` if *mask* is a :class:`~pathlib.Path`.
        * ``None`` if *mask* is ``None`` (no mask entity in filename).
    """
    if mask is None:
        return None
    if isinstance(mask, str):
        return mask
    return "custom"


def _mask_threshold_label(threshold: float) -> str | None:
    """Return a compact string label for a mask threshold value.

    Used to build the ``maskthr-*`` BIDS filename entity when the threshold
    differs from the default of ``0.0``.

    Parameters
    ----------
    threshold
        The mask threshold value (typically in ``[0, 1]``).

    Returns
    -------
    str | None
        * ``None`` when *threshold* is ``0.0`` (default — no entity in filename).
        * An integer-percentage string for "clean" fractions (e.g. ``0.5`` →
          ``"50"``, ``0.25`` → ``"25"``).
        * The raw float representation with ``"."`` replaced by ``"p"`` for
          values that do not map to a whole-number percentage (e.g. ``0.33`` →
          ``"0p33"``).

    Examples
    --------
    >>> _mask_threshold_label(0.0)

    >>> _mask_threshold_label(0.5)
    '50'
    >>> _mask_threshold_label(0.25)
    '25'
    >>> _mask_threshold_label(0.33)
    '0p33'
    """
    if threshold == 0.0:
        return None
    pct = threshold * 100
    if pct == int(pct):
        return str(int(pct))
    return str(threshold).replace(".", "p")


def _atlas_threshold_label(threshold: float) -> str | None:
    """Return a compact string label for an atlas threshold value.

    Used to build the ``atlasthr-*`` BIDS filename entity when the threshold
    differs from the default of ``0.0``.

    Parameters
    ----------
    threshold
        The atlas threshold value (typically in ``[0, 1]``).

    Returns
    -------
    str | None
        * ``None`` when *threshold* is ``0.0`` (default — no entity in filename).
        * An integer-percentage string for "clean" fractions (e.g. ``0.5`` →
          ``"50"``, ``0.25`` → ``"25"``).
        * The raw float representation with ``"."`` replaced by ``"p"`` for
          values that do not map to a whole-number percentage (e.g. ``0.33`` →
          ``"0p33"``).

    Examples
    --------
    >>> _atlas_threshold_label(0.0)

    >>> _atlas_threshold_label(0.5)
    '50'
    >>> _atlas_threshold_label(0.25)
    '25'
    >>> _atlas_threshold_label(0.33)
    '0p33'
    """
    if threshold == 0.0:
        return None
    pct = threshold * 100
    if pct == int(pct):
        return str(int(pct))
    return str(threshold).replace(".", "p")


def _parse_mask(value: str | None) -> Path | str | None:
    """Return the mask as a resolved Path or, for builtin names, as the original string.

    Parameters
    ----------
    value
        The mask value from configuration — either a builtin name (``"gm"``,
        ``"wm"``, ``"brain"``) or a filesystem path.

    Returns
    -------
    Path | str | None
        * ``None`` if *value* is falsy.
        * The original string unchanged if *value* is a builtin mask name.
        * A resolved :class:`~pathlib.Path` otherwise.
    """
    if not value:
        return None
    if value in _BUILTIN_MASK_NAMES:
        return value
    return Path(value).expanduser().resolve()


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


def write_parcellation_sidecar(
    tsv_path: Path,
    original_file: Path,
    atlas_name: str,
    atlas_image: Path,
    atlas_lut: Path | None,
    mask: Path | str | None,
    space: str | None = None,
    resampling_target: str | None = None,
    background_label: int = 0,
    mask_threshold: float = 0.0,
    atlas_threshold: float = 0.0,
) -> Path:
    """Write a JSON sidecar file alongside a parcellation TSV.

    The sidecar captures provenance: which scalar image was parcellated,
    which atlas was used, and what processing parameters were applied.

    Parameters
    ----------
    tsv_path
        Path to the parcellation TSV file. The JSON will share its stem.
    original_file
        Path to the original scalar image that was parcellated.
    atlas_name
        Name of the atlas used.
    atlas_image
        Path to the atlas NIfTI file.
    atlas_lut
        Path to the atlas lookup table, or None.
    mask
        Path to the mask file, the string name of a builtin mask (e.g. "gm"), or None.
    space
        The atlas/data space identifier (e.g. "MNI152NLin2009cAsym").
    resampling_target
        The resampling strategy used (e.g. "data", "labels").
    background_label
        The integer label treated as background.
    mask_threshold
        Threshold applied to the mask image. Voxels with mask values strictly
        greater than this value are included. Default is ``0.0`` (any non-zero
        voxel passes). Recorded as provenance in the sidecar.
    atlas_threshold
        Threshold applied to probabilistic (4D) atlas volumes. Voxels with
        probability strictly greater than this value are included per region.
        Default is ``0.0``. Recorded as provenance in the sidecar.

    Returns
    -------
    Path
        Path to the written JSON sidecar file.
    """
    try:
        from importlib.metadata import version as pkg_version

        software_version = pkg_version("parcellate")
    except Exception:
        software_version = "unknown"

    sidecar: dict = {
        "original_file": str(original_file),
        "mask": str(mask) if mask is not None else None,
        "mask_threshold": mask_threshold,
        "atlas_threshold": atlas_threshold,
        "parcellation_scheme": {
            "name": atlas_name,
            "image": str(atlas_image),
            "lut": str(atlas_lut) if atlas_lut is not None else None,
        },
        "space": space,
        "resampling_target": resampling_target,
        "background_label": background_label,
        "software_version": software_version,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    json_path = tsv_path.with_suffix(".json")
    json_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    logger.debug("Wrote parcellation sidecar to %s", json_path)
    return json_path


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
        atlas_threshold = float(cfg.get("atlas_threshold", 0.0))

        atlases.append(
            AtlasDefinition(
                name=name,
                nifti_path=Path(path).expanduser().resolve(),
                lut=lut_path,
                space=space,
                resolution=resolution,
                atlas_threshold=atlas_threshold,
            )
        )
    return atlases

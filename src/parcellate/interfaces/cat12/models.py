"""Structured representations of CAT12 inputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from parcellate.interfaces.models import (
    AtlasDefinition,
    ParcellationOutput,
    ReconInput,
    SubjectContext,
)


class TissueType(str, Enum):
    """CAT12 tissue type classification."""

    GM = "GM"  # Gray matter (mwp1*)
    WM = "WM"  # White matter (mwp2*)
    CT = "CT"  # Cortical thickness (wct*)


@dataclass(frozen=True)
class ScalarMapDefinition:
    """Description of a scalar map available to the pipeline."""

    name: str
    nifti_path: Path
    tissue_type: TissueType | None = None
    space: str | None = None
    desc: str | None = None


@dataclass
class Cat12Config:
    """Configuration parsed from TOML input."""

    input_root: Path
    output_dir: Path
    atlases: list[AtlasDefinition] | None = None
    subjects: list[str] | None = None
    sessions: list[str] | None = None
    mask: Path | str | None = "gm"
    background_label: int = 0
    resampling_target: str | None = "data"
    force: bool = False
    log_level: int = logging.INFO
    n_jobs: int = 1
    n_procs: int = 1


__all__ = [
    "AtlasDefinition",
    "Cat12Config",
    "ParcellationOutput",
    "ReconInput",
    "ScalarMapDefinition",
    "SubjectContext",
    "TissueType",
]

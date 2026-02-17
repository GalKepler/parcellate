"""Shared structured representations of pipeline inputs and outputs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    ScalarMapDefinition = Any
else:
    ScalarMapDefinition = "ScalarMapDefinition"


@dataclass
class ParcellationConfig:
    """Base configuration for parcellation workflows.

    This dataclass consolidates common configuration fields shared across
    all parcellation interfaces (CAT12, QSIRecon, etc.).
    """

    input_root: Path
    output_dir: Path
    atlases: list[AtlasDefinition] | None = None
    subjects: list[str] | None = None
    sessions: list[str] | None = None
    mask: Path | str | None = None
    mask_threshold: float = 0.0
    background_label: int = 0
    resampling_target: str | None = "data"
    force: bool = False
    log_level: int = logging.INFO
    n_jobs: int = 1
    n_procs: int = 1


@dataclass(frozen=True)
class ScalarMapBase:
    """Base class for scalar map definitions.

    This base class defines the common fields shared across all interfaces.
    Interface-specific subclasses extend this with additional metadata.
    """

    name: str
    nifti_path: Path
    space: str | None = None
    desc: str | None = None


@dataclass(frozen=True)
class SubjectContext:
    """Minimal BIDS-like identifier for a subject/session."""

    subject_id: str
    session_id: str | None = None

    @property
    def label(self) -> str:
        """Return a compact label suitable for filenames."""

        return f"sub-{self.subject_id}" + (f"_ses-{self.session_id}" if self.session_id else "")


@dataclass(frozen=True)
class AtlasDefinition:
    """Description of an atlas available to the pipeline."""

    name: str
    nifti_path: Path
    lut: pd.DataFrame | Path | None = None
    resolution: str | None = None
    space: str | None = None


@dataclass(frozen=True)
class ReconInput:
    """Paths to reconstruction outputs required for parcellation."""

    context: SubjectContext
    atlases: Sequence[AtlasDefinition]
    scalar_maps: Sequence[ScalarMapDefinition]
    transforms: Sequence[Path] = ()


@dataclass(frozen=True)
class ParcellationOutput:
    """Paths to parcellation outputs."""

    context: SubjectContext
    atlas: AtlasDefinition
    scalar_map: ScalarMapDefinition
    stats_table: pd.DataFrame

"""Structured representations of inputs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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
class ScalarMapDefinition:
    """Description of a scalar map available to the pipeline."""

    name: str
    nifti_path: Path
    param: str | None = None
    model: str | None = None
    origin: str | None = None
    space: str | None = None
    desc: str | None = None
    recon_workflow: str | None = None


@dataclass(frozen=True)
class ReconInput:
    """Paths to QSIRecon outputs required for parcellation."""

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


@dataclass
class QSIReconConfig:
    """Configuration parsed from TOML input."""

    input_root: Path
    output_dir: Path
    subjects: list[str] | None = None
    sessions: list[str] | None = None
    mask: Path | None = None
    background_label: int = 0
    resampling_target: str | None = "data"
    force: bool = False
    log_level: int = logging.INFO

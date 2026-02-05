"""Shared structured representations of pipeline inputs and outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    ScalarMapDefinition = Any
else:
    ScalarMapDefinition = "ScalarMapDefinition"


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

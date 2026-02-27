"""Structured representations of CAT12 inputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from parcellate.interfaces.models import (
    AtlasDefinition,
    ParcellationConfig,
    ParcellationOutput,
    ReconInput,
    ScalarMapBase,
    SubjectContext,
)


class AtlasConfigurationError(ValueError):
    """Raised when no atlases are configured for the CAT12 interface."""


class TissueType(str, Enum):
    """CAT12 tissue type classification."""

    GM = "GM"  # Gray matter (mwp1*)
    WM = "WM"  # White matter (mwp2*)
    CT = "CT"  # Cortical thickness (wct*)


@dataclass(frozen=True)
class ScalarMapDefinition(ScalarMapBase):
    """Description of a scalar map available to the pipeline.

    Extends ScalarMapBase with CAT12-specific fields.
    """

    tissue_type: TissueType | None = None


@dataclass
class Cat12Config(ParcellationConfig):
    """Configuration for CAT12 parcellation workflow."""


__all__ = [
    "AtlasConfigurationError",
    "AtlasDefinition",
    "Cat12Config",
    "ParcellationOutput",
    "ReconInput",
    "ScalarMapDefinition",
    "SubjectContext",
    "TissueType",
]

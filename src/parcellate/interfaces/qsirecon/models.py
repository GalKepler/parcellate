"""Structured representations of inputs."""

from __future__ import annotations

from dataclasses import dataclass

from parcellate.interfaces.models import (
    AtlasDefinition,
    ParcellationConfig,
    ParcellationOutput,
    ReconInput,
    ScalarMapBase,
    SubjectContext,
)


@dataclass(frozen=True)
class ScalarMapDefinition(ScalarMapBase):
    """Description of a scalar map available to the pipeline.

    Extends ScalarMapBase with QSIRecon-specific fields.
    """

    param: str | None = None
    model: str | None = None
    origin: str | None = None
    recon_workflow: str | None = None


@dataclass
class QSIReconConfig(ParcellationConfig):
    """Configuration for QSIRecon parcellation workflow.

    Inherits all fields from ParcellationConfig with no overrides.
    """


__all__ = [
    "AtlasDefinition",
    "ParcellationOutput",
    "QSIReconConfig",
    "ReconInput",
    "ScalarMapDefinition",
    "SubjectContext",
]

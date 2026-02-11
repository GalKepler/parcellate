"""CAT12 parcellation interface."""

from parcellate.interfaces.cat12.cat12 import load_config, run_parcellations
from parcellate.interfaces.cat12.cli import (
    SubjectSession,
    config_from_env,
    load_subjects_from_csv,
    process_single_subject,
    run_parcellations_parallel,
)
from parcellate.interfaces.cat12.loader import load_cat12_inputs
from parcellate.interfaces.cat12.models import (
    AtlasDefinition,
    Cat12Config,
    ParcellationOutput,
    ReconInput,
    ScalarMapDefinition,
    SubjectContext,
    TissueType,
)
from parcellate.interfaces.planner import plan_parcellation_workflow as plan_cat12_parcellation_workflow
from parcellate.interfaces.runner import run_parcellation_workflow as run_cat12_parcellation_workflow

__all__ = [
    "AtlasDefinition",
    "Cat12Config",
    "ParcellationOutput",
    "ReconInput",
    "ScalarMapDefinition",
    "SubjectContext",
    "SubjectSession",
    "TissueType",
    "config_from_env",
    "load_cat12_inputs",
    "load_config",
    "load_subjects_from_csv",
    "plan_cat12_parcellation_workflow",
    "process_single_subject",
    "run_cat12_parcellation_workflow",
    "run_parcellations",
    "run_parcellations_parallel",
]

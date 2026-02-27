"""High-level orchestration for parcellating QSIRecon outputs.

This module provides a small CLI that reads a TOML configuration file,
loads recon inputs, plans a parcellation workflow, runs it, and writes the
outputs into a BIDS-derivative-style directory mirroring the structure used
by QSIRecon.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from parcellate.interfaces.models import ParcellationOutput
from parcellate.interfaces.planner import plan_parcellation_workflow
from parcellate.interfaces.qsirecon.loader import load_qsirecon_inputs
from parcellate.interfaces.qsirecon.models import (
    AtlasDefinition,
    QSIReconConfig,
    ReconInput,
    ScalarMapDefinition,
    SubjectContext,
)
from parcellate.interfaces.runner import run_parcellation_workflow
from parcellate.interfaces.shared import (
    add_cli_args,
    build_pending_plan,
    load_config_base,
    run_parallel_workflow,
    write_output,
)
from parcellate.interfaces.utils import (
    _atlas_threshold_label,
    _mask_label,
    _mask_threshold_label,
)

LOGGER = logging.getLogger(__name__)


def load_config(args: argparse.Namespace) -> QSIReconConfig:
    """Parse a TOML configuration file and override with CLI arguments."""
    return load_config_base(args, QSIReconConfig)  # No default atlas space for QSIRecon


def _build_output_path(
    context: SubjectContext,
    atlas: AtlasDefinition,
    scalar_map: ScalarMapDefinition,
    destination: Path,
    mask: Path | str | None = None,
    mask_threshold: float = 0.0,
    atlas_threshold: float = 0.0,
) -> Path:
    """Construct the output path for a QSIRecon parcellation result."""
    workflow = scalar_map.recon_workflow or "parcellate"
    base = destination / f"qsirecon-{workflow}"

    subject_dir = base / f"sub-{context.subject_id}"
    if context.session_id:
        subject_dir = subject_dir / f"ses-{context.session_id}"

    output_dir = subject_dir / "dwi" / f"atlas-{atlas.name}"

    entities: list[str] = [context.label]
    space = atlas.space or scalar_map.space
    entities.append(f"atlas-{atlas.name}")
    if space:
        entities.append(f"space-{space}")
    if atlas.resolution:
        entities.append(f"res-{atlas.resolution}")
    if scalar_map.model:
        entities.append(f"model-{scalar_map.model}")
    entities.append(f"param-{scalar_map.param}")
    if scalar_map.desc:
        entities.append(f"desc-{scalar_map.desc}")
    label = _mask_label(mask)
    if label:
        entities.append(f"mask-{label}")
    thr_label = _mask_threshold_label(mask_threshold)
    if thr_label:
        entities.append(f"maskthr-{thr_label}")
    athr_label = _atlas_threshold_label(atlas_threshold)
    if athr_label:
        entities.append(f"atlasthr-{athr_label}")

    filename = "_".join([*entities, "parc"]) + ".tsv"
    return output_dir / filename


def _write_output(result: ParcellationOutput, destination: Path, config: QSIReconConfig) -> Path:
    """Write a parcellation output and JSON sidecar for a QSIRecon result."""
    return write_output(result, destination=destination, config=config, build_output_path_fn=_build_output_path)


def _run_recon(recon: ReconInput, config: QSIReconConfig) -> list[Path]:
    """Run the parcellation workflow for a single QSIRecon recon."""
    plan = plan_parcellation_workflow(recon)
    pending_plan, reused_outputs = build_pending_plan(
        recon=recon,
        config=config,
        plan=plan,
        build_output_path_fn=_build_output_path,
        catch_value_error=True,  # QSIRecon: param field may be missing
    )
    outputs: list[Path] = []

    if pending_plan:
        jobs = run_parcellation_workflow(recon=recon, plan=pending_plan, config=config)
        for result in jobs:
            outputs.append(
                write_output(
                    result, destination=config.output_dir, config=config, build_output_path_fn=_build_output_path
                )
            )

    outputs.extend(reused_outputs)
    return outputs


def run_parcellations(config: QSIReconConfig) -> list[Path]:
    """Execute the full QSIRecon parcellation workflow from a parsed config."""
    logging.basicConfig(level=config.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    LOGGER.info("Loading QSIRecon inputs from %s", config.input_root)

    recon_inputs = load_qsirecon_inputs(
        root=config.input_root,
        subjects=config.subjects,
        sessions=config.sessions,
        atlases=config.atlases,
        max_workers=config.n_jobs,
    )
    if not recon_inputs:
        LOGGER.warning("No recon inputs discovered. Nothing to do.")
        return []

    return run_parallel_workflow(config, recon_inputs, _run_recon, "QSIRecon")


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI execution."""
    parser = argparse.ArgumentParser(description="Run parcellations for QSIRecon derivatives.")
    add_cli_args(parser, "Root directory of QSIRecon derivatives.")
    args = parser.parse_args(argv)
    config = load_config(args)

    try:
        run_parcellations(config)
    except Exception:  # pragma: no cover - defensive logging for CLI execution
        LOGGER.exception("Parcellation workflow failed")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

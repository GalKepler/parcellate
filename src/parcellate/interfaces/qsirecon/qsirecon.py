"""High-level orchestration for parcellating QSIRecon outputs.

This module provides a small CLI that reads a TOML configuration file,
loads recon inputs, plans a parcellation workflow, runs it, and writes the
outputs into a BIDS-derivative-style directory mirroring the structure used
by QSIRecon.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older environments
    import tomli as tomllib  # type: ignore[import]

from parcellate.interfaces.qsirecon.loader import load_qsirecon_inputs
from parcellate.interfaces.qsirecon.models import (
    AtlasDefinition,
    ParcellationOutput,
    QSIReconConfig,
    ScalarMapDefinition,
    SubjectContext,
)
from parcellate.interfaces.qsirecon.planner import plan_qsirecon_parcellation_workflow
from parcellate.interfaces.qsirecon.runner import run_qsirecon_parcellation_workflow

LOGGER = logging.getLogger(__name__)


def _parse_log_level(value: str | int | None) -> int:
    """Return a logging level from common string/int inputs."""

    if value is None:
        return logging.INFO
    if isinstance(value, int):
        return value
    return getattr(logging, str(value).upper(), logging.INFO)


def load_config(config_path: Path) -> QSIReconConfig:
    """Parse a TOML configuration file.

    The configuration expects the following optional keys:
    - ``input_root``: Root directory of QSIRecon derivatives.
    - ``output_dir``: Destination directory for parcellation outputs.
    - ``subjects``: List of subject identifiers to process.
    - ``sessions``: List of session identifiers to process.
    - ``mask``: Optional path to a brain mask to apply during parcellation.
    - ``force``: Whether to overwrite existing parcellation outputs.
    - ``log_level``: Logging verbosity (e.g., ``INFO``, ``DEBUG``).
    """

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    input_root = Path(data.get("input_root", ".")).expanduser().resolve()
    output_dir = Path(data.get("output_dir", input_root / "parcellations")).expanduser().resolve()
    subjects = _as_list(data.get("subjects"))
    sessions = _as_list(data.get("sessions"))

    mask_value = data.get("mask")
    mask = Path(mask_value).expanduser().resolve() if mask_value else None
    force = bool(data.get("force", False))
    log_level = _parse_log_level(data.get("log_level"))

    return QSIReconConfig(
        input_root=input_root,
        output_dir=output_dir,
        subjects=subjects,
        sessions=sessions,
        mask=mask,
        force=force,
        log_level=log_level,
    )


def _as_list(value: Iterable[str] | str | None) -> list[str] | None:
    """Normalize configuration values into a list of strings."""

    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _build_output_path(
    context: SubjectContext,
    atlas: AtlasDefinition,
    scalar_map: ScalarMapDefinition,
    destination: Path,
) -> Path:
    """Construct the output path for a parcellation result."""

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

    filename = "_".join([*entities, "parc"]) + ".tsv"
    return output_dir / filename


def _write_output(result: ParcellationOutput, destination: Path) -> Path:
    """Write a parcellation output to disk using a QSIRecon-like layout."""

    out_path = _build_output_path(
        context=result.context,
        atlas=result.atlas,
        scalar_map=result.scalar_map,
        destination=destination,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result.stats_table.to_csv(out_path, sep="\t", index=False)
    LOGGER.debug("Wrote parcellation output to %s", out_path)
    return out_path


def run_parcellations(config: QSIReconConfig) -> list[Path]:
    """Execute the full parcellation workflow from a parsed config."""

    logging.basicConfig(level=config.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    LOGGER.info("Loading QSIRecon inputs from %s", config.input_root)

    recon_inputs = load_qsirecon_inputs(
        root=config.input_root,
        subjects=config.subjects,
        sessions=config.sessions,
    )
    if not recon_inputs:
        LOGGER.warning("No recon inputs discovered. Nothing to do.")
        return []

    outputs: list[Path] = []
    for recon in recon_inputs:
        plan = plan_qsirecon_parcellation_workflow(recon)
        pending_plan: dict[AtlasDefinition, list[ScalarMapDefinition]] = {}
        reused_outputs: list[Path] = []

        for atlas, scalar_maps in plan.items():
            remaining: list[ScalarMapDefinition] = []
            for scalar_map in scalar_maps:
                out_path = _build_output_path(
                    context=recon.context,
                    atlas=atlas,
                    scalar_map=scalar_map,
                    destination=config.output_dir,
                )
                if not config.force and out_path.exists():
                    LOGGER.info("Reusing existing parcellation output at %s", out_path)
                    _ = pd.read_csv(out_path, sep="\t")
                    reused_outputs.append(out_path)
                else:
                    remaining.append(scalar_map)
            if remaining:
                pending_plan[atlas] = remaining

        if pending_plan:
            jobs = run_qsirecon_parcellation_workflow(recon=recon, plan=pending_plan, config=config)
            for result in jobs:
                outputs.append(_write_output(result, destination=config.output_dir))

        outputs.extend(reused_outputs)
    LOGGER.info("Finished writing %d parcellation files", len(outputs))
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run parcellations for QSIRecon derivatives.")
    parser.add_argument(
        "config",
        type=Path,
        help="Path to a TOML configuration file describing inputs and outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI execution."""

    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)

    try:
        run_parcellations(config)
    except Exception:  # pragma: no cover - defensive logging for CLI execution
        LOGGER.exception("Parcellation workflow failed")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

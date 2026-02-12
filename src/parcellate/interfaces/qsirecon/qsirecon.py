"""High-level orchestration for parcellating QSIRecon outputs.

This module provides a small CLI that reads a TOML configuration file,
loads recon inputs, plans a parcellation workflow, runs it, and writes the
outputs into a BIDS-derivative-style directory mirroring the structure used
by QSIRecon.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older environments
    import tomli as tomllib  # type: ignore[import]

from parcellate.interfaces.planner import plan_parcellation_workflow
from parcellate.interfaces.qsirecon.loader import load_qsirecon_inputs
from parcellate.interfaces.qsirecon.models import (
    AtlasDefinition,
    ParcellationOutput,
    QSIReconConfig,
    ReconInput,
    ScalarMapDefinition,
    SubjectContext,
)
from parcellate.interfaces.runner import run_parcellation_workflow
from parcellate.interfaces.utils import _as_list, _parse_log_level, parse_atlases

LOGGER = logging.getLogger(__name__)


def load_config(args: argparse.Namespace) -> QSIReconConfig:
    """Parse a TOML configuration file and override with CLI arguments."""

    data = {}
    if args.config:
        with args.config.open("rb") as f:
            data = tomllib.load(f)

    input_root_str = args.input_root or data.get("input_root", ".")
    input_root = Path(input_root_str).expanduser().resolve()
    output_dir_str = args.output_dir or data.get("output_dir", input_root / "parcellations")
    output_dir = Path(output_dir_str).expanduser().resolve()
    subjects = args.subjects or _as_list(data.get("subjects"))
    sessions = args.sessions or _as_list(data.get("sessions"))

    atlas_configs = []
    if args.atlas_config:
        for atlas_path in args.atlas_config:
            with atlas_path.open("rb") as f:
                atlas_configs.append(tomllib.load(f))
    else:
        atlas_configs = data.get("atlases", [])
    atlases = parse_atlases(atlas_configs)  # No default space for QSIRecon

    mask_value = args.mask or data.get("mask")
    mask = Path(mask_value).expanduser().resolve() if mask_value else None
    force = args.force or bool(data.get("force", False))
    log_level = _parse_log_level(args.log_level or data.get("log_level"))
    n_jobs = args.n_jobs or int(data.get("n_jobs", 1))
    n_procs = args.n_procs or int(data.get("n_procs", 1))

    return QSIReconConfig(
        input_root=input_root,
        output_dir=output_dir,
        atlases=atlases,
        subjects=subjects,
        sessions=sessions,
        mask=mask,
        force=force,
        log_level=log_level,
        n_jobs=n_jobs,
        n_procs=n_procs,
    )


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


def _run_recon(
    recon: ReconInput,
    plan: dict[AtlasDefinition, list[ScalarMapDefinition]],
    config: QSIReconConfig,
) -> list[Path]:
    """Run the parcellation workflow for a single recon."""
    pending_plan: dict[AtlasDefinition, list[ScalarMapDefinition]] = {}
    reused_outputs: list[Path] = []
    outputs: list[Path] = []

    for atlas, scalar_maps in plan.items():
        remaining: list[ScalarMapDefinition] = []
        for scalar_map in scalar_maps:
            try:
                out_path = _build_output_path(
                    context=recon.context,
                    atlas=atlas,
                    scalar_map=scalar_map,
                    destination=config.output_dir,
                )
                if not config.force and out_path.exists():
                    LOGGER.info("Reusing existing parcellation output at %s", out_path)
                    reused_outputs.append(out_path)
                else:
                    remaining.append(scalar_map)
            except ValueError:
                LOGGER.warning(
                    "Could not build output path for %s, %s, %s",
                    recon.context,
                    atlas,
                    scalar_map,
                )
                continue
        if remaining:
            pending_plan[atlas] = remaining

    if pending_plan:
        jobs = run_parcellation_workflow(recon=recon, plan=pending_plan, config=config)
        for result in jobs:
            outputs.append(_write_output(result, destination=config.output_dir))

    outputs.extend(reused_outputs)
    return outputs


def run_parcellations(config: QSIReconConfig) -> list[Path]:
    """Execute the full parcellation workflow from a parsed config."""

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

    if config.n_procs > 1:
        LOGGER.info("Running parcellation across subjects with %d processes", config.n_procs)
        with ProcessPoolExecutor(max_workers=config.n_procs) as executor:
            futures = [
                executor.submit(
                    _run_recon,
                    recon,
                    plan_parcellation_workflow(recon),
                    config,
                )
                for recon in recon_inputs
            ]
            LOGGER.info("Submitted %d parcellation jobs to the executor", len(futures))
            outputs = [future.result() for future in futures]
    else:
        outputs = [_run_recon(recon, plan_parcellation_workflow(recon), config) for recon in recon_inputs]

    flat_outputs = [item for sublist in outputs for item in sublist]
    LOGGER.info("Finished writing %d parcellation files", len(flat_outputs))
    return flat_outputs


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments to the parser."""

    parser.add_argument(
        "--input-root",
        type=Path,
        help="Root directory of QSIRecon derivatives.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory for parcellation outputs.",
    )
    parser.add_argument(
        "--atlas-config",
        type=Path,
        nargs="+",
        help="Path to one or more TOML files defining atlases.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="List of subject identifiers to process.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        help="List of session identifiers to process.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        help="Optional path to a brain mask to apply during parcellation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Whether to overwrite existing parcellation outputs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (e.g., INFO, DEBUG).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="Number of parallel jobs for within-subject parcellation.",
    )
    parser.add_argument(
        "--n-procs",
        type=int,
        help="Number of parallel processes for across-subject parcellation.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a TOML configuration file.",
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI execution."""

    parser = argparse.ArgumentParser(description="Run parcellations for QSIRecon derivatives.")
    add_cli_args(parser)
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

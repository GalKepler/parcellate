"""High-level orchestration for parcellating CAT12 outputs.

This module provides a small CLI that reads a TOML configuration file,
loads CAT12 inputs, plans a parcellation workflow, runs it, and writes the
outputs into a BIDS-derivative-style directory.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older environments
    import tomli as tomllib  # type: ignore[import]

from parcellate.interfaces.cat12.loader import load_cat12_inputs
from parcellate.interfaces.cat12.models import (
    AtlasDefinition,
    Cat12Config,
    ParcellationOutput,
    ReconInput,
    ScalarMapDefinition,
    SubjectContext,
)
from parcellate.interfaces.cat12.planner import plan_cat12_parcellation_workflow
from parcellate.interfaces.cat12.runner import run_cat12_parcellation_workflow
from parcellate.interfaces.utils import _as_list, _parse_log_level

LOGGER = logging.getLogger(__name__)


def load_config(config_path: Path) -> Cat12Config:
    """Parse a TOML configuration file.

    The configuration expects the following keys:
    - ``input_root``: Root directory of CAT12 derivatives.
    - ``output_dir``: Destination directory for parcellation outputs.
    - ``atlases``: List of atlas definitions (each with name, path, lut, space).
    - ``subjects``: Optional list of subject identifiers to process.
    - ``sessions``: Optional list of session identifiers to process.
    - ``mask``: Optional path to a brain mask to apply during parcellation.
    - ``force``: Whether to overwrite existing parcellation outputs.
    - ``log_level``: Logging verbosity (e.g., ``INFO``, ``DEBUG``).
    - ``n_jobs``: Number of parallel jobs for within-subject parcellation.
    - ``n_procs``: Number of parallel processes for across-subject parcellation.
    """

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    input_root = Path(data.get("input_root", ".")).expanduser().resolve()
    output_dir = Path(data.get("output_dir", input_root / "parcellations")).expanduser().resolve()
    subjects = _as_list(data.get("subjects"))
    sessions = _as_list(data.get("sessions"))

    # Parse atlas definitions
    atlases = _parse_atlases(data.get("atlases", []))

    mask_value = data.get("mask")
    mask = Path(mask_value).expanduser().resolve() if mask_value else None
    force = bool(data.get("force", False))
    log_level = _parse_log_level(data.get("log_level"))
    n_jobs = int(data.get("n_jobs", 1))
    n_procs = int(data.get("n_procs", 1))

    return Cat12Config(
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


def _parse_atlases(atlas_configs: list[dict]) -> list[AtlasDefinition]:
    """Parse atlas definitions from configuration.

    Parameters
    ----------
    atlas_configs
        List of atlas configuration dictionaries.

    Returns
    -------
    list[AtlasDefinition]
        List of parsed atlas definitions.
    """
    atlases = []
    for cfg in atlas_configs:
        name = cfg.get("name")
        path = cfg.get("path")
        if not name or not path:
            LOGGER.warning("Skipping atlas with missing name or path: %s", cfg)
            continue
        lut = cfg.get("lut")
        lut_path = Path(lut).expanduser().resolve() if lut else None
        space = cfg.get("space", "MNI152NLin2009cAsym")
        resolution = cfg.get("resolution")

        atlases.append(
            AtlasDefinition(
                name=name,
                nifti_path=Path(path).expanduser().resolve(),
                lut=lut_path,
                space=space,
                resolution=resolution,
            )
        )
    return atlases


def _build_output_path(
    context: SubjectContext,
    atlas: AtlasDefinition,
    scalar_map: ScalarMapDefinition,
    destination: Path,
) -> Path:
    """Construct the output path for a parcellation result."""

    base = destination / "cat12"

    subject_dir = base / f"sub-{context.subject_id}"
    if context.session_id:
        subject_dir = subject_dir / f"ses-{context.session_id}"

    output_dir = subject_dir / "anat" / f"atlas-{atlas.name}"

    entities: list[str] = [context.label]
    space = atlas.space or scalar_map.space
    entities.append(f"atlas-{atlas.name}")
    if space:
        entities.append(f"space-{space}")
    if atlas.resolution:
        entities.append(f"res-{atlas.resolution}")
    if scalar_map.tissue_type:
        entities.append(f"tissue-{scalar_map.tissue_type.value}")

    filename = "_".join([*entities, "parc"]) + ".tsv"
    return output_dir / filename


def _write_output(result: ParcellationOutput, destination: Path) -> Path:
    """Write a parcellation output to disk using a CAT12-like layout."""

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


def _run_recon(recon: ReconInput, config: Cat12Config) -> list[Path]:
    """Run the parcellation workflow for a single recon."""
    plan = plan_cat12_parcellation_workflow(recon)
    pending_plan: dict[AtlasDefinition, list[ScalarMapDefinition]] = {}
    reused_outputs: list[Path] = []
    outputs: list[Path] = []

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
        jobs = run_cat12_parcellation_workflow(recon=recon, plan=pending_plan, config=config)
        for result in jobs:
            outputs.append(_write_output(result, destination=config.output_dir))

    outputs.extend(reused_outputs)
    return outputs


def run_parcellations(config: Cat12Config) -> list[Path]:
    """Execute the full parcellation workflow from a parsed config."""

    logging.basicConfig(level=config.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    LOGGER.info("Loading CAT12 inputs from %s", config.input_root)

    recon_inputs = load_cat12_inputs(
        root=config.input_root,
        atlases=config.atlases,
        subjects=config.subjects,
        sessions=config.sessions,
    )
    if not recon_inputs:
        LOGGER.warning("No CAT12 inputs discovered. Nothing to do.")
        return []

    if config.n_procs > 1:
        with ProcessPoolExecutor(max_workers=config.n_procs) as executor:
            futures = [executor.submit(_run_recon, recon, config) for recon in recon_inputs]
            outputs = [future.result() for future in futures]
    else:
        outputs = [_run_recon(recon, config) for recon in recon_inputs]

    flat_outputs = [item for sublist in outputs for item in sublist]
    LOGGER.info("Finished writing %d parcellation files", len(flat_outputs))
    return flat_outputs


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run parcellations for CAT12 derivatives.")
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

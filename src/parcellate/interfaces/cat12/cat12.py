"""High-level orchestration for parcellating CAT12 outputs.

This module provides a small CLI that reads a TOML configuration file,
loads CAT12 inputs, plans a parcellation workflow, runs it, and writes the
outputs into a BIDS-derivative-style directory.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from parcellate.interfaces.cat12.loader import discover_cat12_xml, extract_tiv_from_xml, load_cat12_inputs
from parcellate.interfaces.cat12.models import (
    AtlasDefinition,
    Cat12Config,
    ReconInput,
    ScalarMapDefinition,
    SubjectContext,
)
from parcellate.interfaces.models import ParcellationOutput
from parcellate.interfaces.planner import plan_parcellation_workflow
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


def load_config(args: argparse.Namespace) -> Cat12Config:
    """Parse a TOML configuration file and override with CLI arguments.

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
    return load_config_base(args, Cat12Config, default_atlas_space="MNI152NLin2009cAsym")


def _build_output_path(
    context: SubjectContext,
    atlas: AtlasDefinition,
    scalar_map: ScalarMapDefinition,
    destination: Path,
    mask: Path | str | None = None,
    mask_threshold: float = 0.0,
    atlas_threshold: float = 0.0,
) -> Path:
    """Construct the output path for a CAT12 parcellation result."""
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


def _find_first_tiv(root: Path, subject: str, session: str | None) -> float | None:
    """Return the first valid TIV value found in CAT12 XML reports for a subject/session."""
    for xml_path in discover_cat12_xml(root=root, subject=subject, session=session):
        tiv = extract_tiv_from_xml(xml_path)
        if tiv is not None:
            return tiv
    return None


def _build_tiv_output_path(context: SubjectContext, destination: Path) -> Path:
    """Return the output path for a standalone TIV TSV file."""
    base = destination / "cat12"
    subject_dir = base / f"sub-{context.subject_id}"
    if context.session_id:
        subject_dir = subject_dir / f"ses-{context.session_id}"
    return subject_dir / "anat" / f"{context.label}_tiv.tsv"


def _extract_and_write_tiv(root: Path, context: SubjectContext, destination: Path) -> Path | None:
    """Discover CAT12 XML reports, extract TIV values, and write a standalone TSV.

    Parameters
    ----------
    root
        Root directory of CAT12 derivatives.
    context
        Subject/session context.
    destination
        Output root directory.

    Returns
    -------
    Path | None
        Path to the written TSV, or ``None`` if no TIV data was found.
    """
    xml_paths = discover_cat12_xml(root=root, subject=context.subject_id, session=context.session_id)
    if not xml_paths:
        return None
    rows = []
    for xml_path in xml_paths:
        tiv = extract_tiv_from_xml(xml_path)
        if tiv is not None:
            rows.append({"source_file": str(xml_path), "vol_TIV": tiv})
    if not rows:
        return None
    tiv_path = _build_tiv_output_path(context, destination)
    tiv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tiv_path, sep="\t", index=False)
    LOGGER.debug("Wrote TIV output to %s", tiv_path)
    return tiv_path


def _write_output(result: ParcellationOutput, destination: Path, config: Cat12Config) -> Path:
    """Write a parcellation output and JSON sidecar for a CAT12 result."""
    return write_output(result, destination=destination, config=config, build_output_path_fn=_build_output_path)


def _run_recon(recon: ReconInput, config: Cat12Config) -> list[Path]:
    """Run the parcellation workflow for a single CAT12 recon."""
    plan = plan_parcellation_workflow(recon)
    pending_plan, reused_outputs = build_pending_plan(
        recon=recon,
        config=config,
        plan=plan,
        build_output_path_fn=_build_output_path,
    )
    outputs: list[Path] = []

    if pending_plan:
        tiv_value = _find_first_tiv(config.input_root, recon.context.subject_id, recon.context.session_id)
        jobs = run_parcellation_workflow(recon=recon, plan=pending_plan, config=config)
        for result in jobs:
            if tiv_value is not None:
                result.stats_table["vol_TIV"] = tiv_value
            outputs.append(
                write_output(
                    result, destination=config.output_dir, config=config, build_output_path_fn=_build_output_path
                )
            )
        tiv_path = _extract_and_write_tiv(root=config.input_root, context=recon.context, destination=config.output_dir)
        if tiv_path is not None:
            outputs.append(tiv_path)

    outputs.extend(reused_outputs)
    return outputs


def run_parcellations(config: Cat12Config) -> list[Path]:
    """Execute the full CAT12 parcellation workflow from a parsed config."""
    logging.basicConfig(level=config.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    LOGGER.info("Loading CAT12 inputs from %s", config.input_root)

    recon_inputs = load_cat12_inputs(
        root=config.input_root,
        atlases=config.atlases,
        subjects=config.subjects,
        sessions=config.sessions,
        max_workers=config.n_jobs,
    )
    if not recon_inputs:
        LOGGER.warning("No CAT12 inputs discovered. Nothing to do.")
        return []

    return run_parallel_workflow(config, recon_inputs, _run_recon, "CAT12")


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI execution."""
    parser = argparse.ArgumentParser(description="Run parcellations for CAT12 derivatives.")
    add_cli_args(parser, "Root directory of CAT12 derivatives.")
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

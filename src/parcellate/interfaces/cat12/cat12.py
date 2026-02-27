"""High-level orchestration for parcellating CAT12 outputs.

This module provides a small CLI that reads a TOML configuration file,
loads CAT12 inputs, plans a parcellation workflow, runs it, and writes the
outputs into a BIDS-derivative-style directory.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older environments
    import tomli as tomllib  # type: ignore[import]

from parcellate.interfaces.cat12.loader import discover_cat12_xml, extract_tiv_from_xml, load_cat12_inputs
from parcellate.interfaces.cat12.models import (
    AtlasDefinition,
    Cat12Config,
    ParcellationOutput,
    ReconInput,
    ScalarMapDefinition,
    SubjectContext,
)
from parcellate.interfaces.planner import plan_parcellation_workflow
from parcellate.interfaces.runner import run_parcellation_workflow
from parcellate.interfaces.utils import (
    _as_list,
    _atlas_threshold_label,
    _mask_label,
    _mask_threshold_label,
    _parse_log_level,
    _parse_mask,
    parse_atlases,
    write_parcellation_sidecar,
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

    # Parse atlas definitions
    atlas_configs = []
    if args.atlas_config:
        for atlas_path in args.atlas_config:
            with atlas_path.open("rb") as f:
                atlas_configs.append(tomllib.load(f))
    else:
        atlas_configs = data.get("atlases", [])
    atlases = parse_atlases(atlas_configs, default_space="MNI152NLin2009cAsym")

    mask_value = args.mask or data.get("mask")
    mask = _parse_mask(mask_value)
    mask_threshold = float(getattr(args, "mask_threshold", None) or data.get("mask_threshold", 0.0))
    force = args.force or bool(data.get("force", False))
    log_level = _parse_log_level(args.log_level or data.get("log_level"))
    n_jobs = args.n_jobs or int(data.get("n_jobs", 1))
    n_procs = args.n_procs or int(data.get("n_procs", 1))
    stat_tier = getattr(args, "stat_tier", None) or data.get("stat_tier") or None

    return Cat12Config(
        input_root=input_root,
        output_dir=output_dir,
        atlases=atlases,
        subjects=subjects,
        sessions=sessions,
        mask=mask,
        mask_threshold=mask_threshold,
        force=force,
        log_level=log_level,
        n_jobs=n_jobs,
        n_procs=n_procs,
        stat_tier=stat_tier,
    )


def _build_output_path(
    context: SubjectContext,
    atlas: AtlasDefinition,
    scalar_map: ScalarMapDefinition,
    destination: Path,
    mask: Path | str | None = None,
    mask_threshold: float = 0.0,
    atlas_threshold: float = 0.0,
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


def _write_output(result: ParcellationOutput, destination: Path, config: Cat12Config) -> Path:
    """Write a parcellation output and JSON sidecar to disk using a CAT12-like layout."""

    out_path = _build_output_path(
        context=result.context,
        atlas=result.atlas,
        scalar_map=result.scalar_map,
        destination=destination,
        mask=config.mask,
        mask_threshold=config.mask_threshold,
        atlas_threshold=result.atlas.atlas_threshold,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result.stats_table.to_csv(out_path, sep="\t", index=False)
    LOGGER.debug("Wrote parcellation output to %s", out_path)

    lut_path = result.atlas.lut if isinstance(result.atlas.lut, Path) else None
    write_parcellation_sidecar(
        tsv_path=out_path,
        original_file=result.scalar_map.nifti_path,
        atlas_name=result.atlas.name,
        atlas_image=result.atlas.nifti_path,
        atlas_lut=lut_path,
        mask=config.mask,
        space=result.atlas.space or result.scalar_map.space,
        resampling_target=config.resampling_target,
        background_label=config.background_label,
        mask_threshold=config.mask_threshold,
        atlas_threshold=result.atlas.atlas_threshold,
    )
    return out_path


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


def _run_recon(recon: ReconInput, config: Cat12Config) -> list[Path]:
    """Run the parcellation workflow for a single recon."""
    plan = plan_parcellation_workflow(recon)
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
                mask=config.mask,
                mask_threshold=config.mask_threshold,
                atlas_threshold=atlas.atlas_threshold,
            )
            if not config.force and out_path.exists():
                LOGGER.info("Reusing existing parcellation output at %s", out_path)
                reused_outputs.append(out_path)
            else:
                remaining.append(scalar_map)
        if remaining:
            pending_plan[atlas] = remaining

    if pending_plan:
        tiv_value = _find_first_tiv(config.input_root, recon.context.subject_id, recon.context.session_id)
        jobs = run_parcellation_workflow(recon=recon, plan=pending_plan, config=config)
        for result in jobs:
            if tiv_value is not None:
                result.stats_table["vol_TIV"] = tiv_value
            outputs.append(_write_output(result, destination=config.output_dir, config=config))
        tiv_path = _extract_and_write_tiv(root=config.input_root, context=recon.context, destination=config.output_dir)
        if tiv_path is not None:
            outputs.append(tiv_path)

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
        max_workers=config.n_jobs,
    )
    if not recon_inputs:
        LOGGER.warning("No CAT12 inputs discovered. Nothing to do.")
        return []

    outputs: list[Path] = []
    total = len(recon_inputs)
    if config.n_procs > 1:
        LOGGER.info("Running parcellation across subjects with %d processes", config.n_procs)
        with ProcessPoolExecutor(max_workers=config.n_procs) as executor:
            future_to_recon = {executor.submit(_run_recon, recon, config): recon for recon in recon_inputs}
            for i, future in enumerate(as_completed(future_to_recon), start=1):
                recon = future_to_recon[future]
                try:
                    result = future.result()
                    outputs.extend(result)
                    LOGGER.info(
                        "[%d/%d] Finished parcellation for %s (%d outputs)",
                        i,
                        total,
                        recon.context.label,
                        len(result),
                    )
                except Exception:
                    LOGGER.exception(
                        "[%d/%d] Failed parcellation for %s",
                        i,
                        total,
                        recon.context.label,
                    )
    else:
        for i, recon in enumerate(recon_inputs, start=1):
            try:
                result = _run_recon(recon, config)
                outputs.extend(result)
                LOGGER.info(
                    "[%d/%d] Finished parcellation for %s (%d outputs)",
                    i,
                    total,
                    recon.context.label,
                    len(result),
                )
            except Exception:
                LOGGER.exception(
                    "[%d/%d] Failed parcellation for %s",
                    i,
                    total,
                    recon.context.label,
                )

    LOGGER.info("Finished writing %d parcellation files", len(outputs))
    return outputs


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments to the parser."""

    parser.add_argument(
        "--input-root",
        type=Path,
        help="Root directory of CAT12 derivatives.",
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
        "--mask-threshold",
        type=float,
        default=0.0,
        dest="mask_threshold",
        help=(
            "Threshold for the mask image. Voxels with mask values strictly greater than this "
            "value are included; all others are excluded. Default: 0.0 (any non-zero voxel passes). "
            "Use values in [0, 1] for probability maps, e.g. 0.5 for >50%% probability."
        ),
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
    parser.add_argument(
        "--stat-tier",
        dest="stat_tier",
        choices=["core", "extended", "diagnostic", "all"],
        default=None,
        help=(
            "Named statistics tier to compute. "
            "'core': mean/std/median/volume/voxel_count/sum (fastest). "
            "'extended': core + robust estimates and shape descriptors. "
            "'diagnostic'/'all': all 45 built-in statistics (default)."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI execution."""

    parser = argparse.ArgumentParser(description="Run parcellations for CAT12 derivatives.")
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

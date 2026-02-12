"""Command-line interface for CAT12 parcellation.

This module provides a CLI that can:
- Read subject/session information from a CSV file
- Load configuration from environment variables or .env file
- Run parcellations in parallel for faster processing
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

from parcellate.interfaces.cat12.loader import discover_scalar_maps
from parcellate.interfaces.cat12.models import (
    AtlasDefinition,
    Cat12Config,
    ParcellationOutput,
    ReconInput,
    SubjectContext,
)
from parcellate.interfaces.planner import plan_parcellation_workflow
from parcellate.interfaces.runner import run_parcellation_workflow

LOGGER = logging.getLogger(__name__)

# Environment variable names
ENV_CAT12_ROOT = "CAT12_ROOT"
ENV_OUTPUT_DIR = "CAT12_OUTPUT_DIR"
ENV_ATLAS_PATHS = "CAT12_ATLAS_PATHS"
ENV_ATLAS_NAMES = "CAT12_ATLAS_NAMES"
ENV_ATLAS_LUTS = "CAT12_ATLAS_LUTS"
ENV_ATLAS_SPACE = "CAT12_ATLAS_SPACE"
ENV_MASK = "CAT12_MASK"
ENV_WORKERS = "CAT12_WORKERS"
ENV_LOG_LEVEL = "CAT12_LOG_LEVEL"


def _load_dotenv() -> None:
    """Load environment variables from .env file if available."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        LOGGER.debug("python-dotenv not installed, skipping .env file loading")


@dataclass
class SubjectSession:
    """A subject/session pair from CSV input."""

    subject_code: str
    session_id: str | None


def sanitize_subject_code(code: str) -> str:
    """Sanitize subject code by stripping whitespace."""
    return code.replace("-", "").replace("_", "").replace(" ", "").zfill(4)


def sanitize_session_id(session: str | int | float) -> str | None:
    """Sanitize session ID by converting to string and stripping whitespace."""
    if isinstance(session, float):
        if pd.isna(session):
            return None
        session = str(int(session))
    elif isinstance(session, int):
        session = str(session)
    return session


def load_subjects_from_csv(csv_path: Path) -> list[SubjectSession]:
    """Load subject/session pairs from a CSV file.

    The CSV file must have a 'subject_code' column and optionally a 'session_id' column.

    Parameters
    ----------
    csv_path
        Path to the CSV file.

    Returns
    -------
    list[SubjectSession]
        List of subject/session pairs.

    Raises
    ------
    ValueError
        If the CSV file is missing the required 'subject_code' column.
    """
    # Read all columns as strings to preserve leading zeros
    df = pd.read_csv(csv_path, dtype=str)

    if "subject_code" not in df.columns:
        msg = f"CSV file must have a 'subject_code' column. Found columns: {list(df.columns)}"
        raise ValueError(msg)

    df["subject_code"] = df["subject_code"].apply(sanitize_subject_code)
    if "session_id" in df.columns:
        df["session_id"] = df["session_id"].apply(sanitize_session_id)

    subjects = []
    for _, row in df.iterrows():
        subject_code = row["subject_code"]
        session_id = (
            row["session_id"]
            if "session_id" in df.columns and pd.notna(row.get("session_id")) and row.get("session_id")
            else None
        )
        subjects.append(SubjectSession(subject_code=subject_code, session_id=session_id))

    return subjects


def _parse_atlases_from_env() -> list[AtlasDefinition]:
    """Parse atlas definitions from environment variables.

    Environment variables:
    - CAT12_ATLAS_PATHS: Comma-separated list of atlas NIfTI paths
    - CAT12_ATLAS_NAMES: Comma-separated list of atlas names (optional)
    - CAT12_ATLAS_LUTS: Comma-separated list of atlas LUT paths (optional)
    - CAT12_ATLAS_SPACE: Space for all atlases (default: MNI152NLin2009cAsym)

    Returns
    -------
    list[AtlasDefinition]
        List of atlas definitions.
    """
    atlas_paths_str = os.getenv(ENV_ATLAS_PATHS, "")
    if not atlas_paths_str:
        return []

    atlas_paths = [Path(p.strip()).expanduser().resolve() for p in atlas_paths_str.split(",") if p.strip()]
    atlas_names_str = os.getenv(ENV_ATLAS_NAMES, "")
    atlas_names = [n.strip() for n in atlas_names_str.split(",") if n.strip()] if atlas_names_str else []
    atlas_luts_str = os.getenv(ENV_ATLAS_LUTS, "")
    atlas_luts = (
        [Path(p.strip()).expanduser().resolve() for p in atlas_luts_str.split(",") if p.strip()]
        if atlas_luts_str
        else []
    )
    space = os.getenv(ENV_ATLAS_SPACE, "MNI152NLin2009cAsym")

    atlases = []
    for i, path in enumerate(atlas_paths):
        name = atlas_names[i] if i < len(atlas_names) else path.stem
        lut = atlas_luts[i] if i < len(atlas_luts) else None
        atlases.append(
            AtlasDefinition(
                name=name,
                nifti_path=path,
                lut=lut,
                space=space,
            )
        )

    return atlases


def config_from_env() -> Cat12Config:
    """Create a Cat12Config from environment variables.

    Environment variables:
    - CAT12_ROOT: Root directory of CAT12 derivatives (required)
    - CAT12_OUTPUT_DIR: Output directory for parcellation results
    - CAT12_ATLAS_PATHS: Comma-separated list of atlas NIfTI paths
    - CAT12_ATLAS_NAMES: Comma-separated list of atlas names
    - CAT12_ATLAS_LUTS: Comma-separated list of atlas LUT paths
    - CAT12_ATLAS_SPACE: Space for all atlases (default: MNI152NLin2009cAsym)
    - CAT12_MASK: Path to brain mask (optional)
    - CAT12_LOG_LEVEL: Logging level (default: INFO)

    Returns
    -------
    Cat12Config
        Configuration object.

    Raises
    ------
    ValueError
        If CAT12_ROOT is not set.
    """
    input_root_str = os.getenv(ENV_CAT12_ROOT)
    if not input_root_str:
        msg = f"Environment variable {ENV_CAT12_ROOT} must be set"
        raise ValueError(msg)

    input_root = Path(input_root_str).expanduser().resolve()
    output_dir_str = os.getenv(ENV_OUTPUT_DIR)
    output_dir = Path(output_dir_str).expanduser().resolve() if output_dir_str else input_root / "parcellations"

    mask_str = os.getenv(ENV_MASK)
    mask = Path(mask_str).expanduser().resolve() if mask_str else None

    atlases = _parse_atlases_from_env()
    log_level = _parse_log_level(os.getenv(ENV_LOG_LEVEL))

    return Cat12Config(
        input_root=input_root,
        output_dir=output_dir,
        atlases=atlases,
        mask=mask,
        log_level=log_level,
    )


def _build_output_path(
    context: SubjectContext,
    atlas: AtlasDefinition,
    scalar_map,
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
    """Write a parcellation output to disk."""
    out_path = _build_output_path(
        context=result.context,
        atlas=result.atlas,
        scalar_map=result.scalar_map,
        destination=destination,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.stats_table.to_csv(out_path, sep="\t", index=False)
    return out_path


def process_single_subject(
    subject: SubjectSession,
    config: Cat12Config,
) -> list[Path]:
    """Process a single subject/session.

    Parameters
    ----------
    subject
        Subject/session pair to process.
    config
        Configuration object.

    Returns
    -------
    list[Path]
        List of output file paths.
    """
    context = SubjectContext(
        subject_id=subject.subject_code,
        session_id=subject.session_id,
    )

    scalar_maps = discover_scalar_maps(
        root=config.input_root,
        subject=subject.subject_code,
        session=subject.session_id,
    )

    if not scalar_maps:
        LOGGER.warning("No scalar maps found for %s", context.label)
        return []

    recon = ReconInput(
        context=context,
        atlases=config.atlases or [],
        scalar_maps=scalar_maps,
    )

    # Check for existing outputs and filter
    plan = plan_parcellation_workflow(recon)
    pending_plan = {}
    reused_outputs: list[Path] = []

    for atlas, maps in plan.items():
        remaining = []
        for scalar_map in maps:
            out_path = _build_output_path(context, atlas, scalar_map, config.output_dir)
            if not config.force and out_path.exists():
                LOGGER.debug("Reusing existing output: %s", out_path)
                reused_outputs.append(out_path)
            else:
                remaining.append(scalar_map)
        if remaining:
            pending_plan[atlas] = remaining

    outputs: list[Path] = []
    if pending_plan:
        results = run_parcellation_workflow(recon=recon, plan=pending_plan, config=config)
        for result in results:
            out_path = _write_output(result, config.output_dir)
            outputs.append(out_path)
            LOGGER.info("Wrote: %s", out_path)

    outputs.extend(reused_outputs)
    return outputs


def _process_subject_wrapper(
    args: tuple[SubjectSession, Cat12Config],
) -> tuple[str, list[Path], str | None]:
    """Wrapper for process_single_subject for use with ProcessPoolExecutor.

    Returns tuple of (subject_label, outputs, error_message).
    """
    subject, config = args
    label = f"sub-{subject.subject_code}" + (f"_ses-{subject.session_id}" if subject.session_id else "")
    try:
        outputs = process_single_subject(subject, config)
        return (label, outputs, None)
    except Exception as e:
        return (label, [], str(e))


def run_parcellations_parallel(
    subjects: Sequence[SubjectSession],
    config: Cat12Config,
    max_workers: int | None = None,
) -> dict[str, list[Path]]:
    """Run parcellations for multiple subjects in parallel.

    Parameters
    ----------
    subjects
        List of subject/session pairs to process.
    config
        Configuration object.
    max_workers
        Maximum number of parallel workers. Defaults to number of CPUs.

    Returns
    -------
    dict[str, list[Path]]
        Dictionary mapping subject labels to their output files.
    """
    if max_workers is None:
        workers_str = os.getenv(ENV_WORKERS)
        max_workers = int(workers_str) if workers_str else None

    results: dict[str, list[Path]] = {}
    errors: dict[str, str] = {}

    # Prepare arguments for parallel processing
    args_list = [(subject, config) for subject in subjects]

    total = len(subjects)
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_subject_wrapper, args): args[0] for args in args_list}

        for future in as_completed(futures):
            subject = futures[future]
            label = f"sub-{subject.subject_code}" + (f"_ses-{subject.session_id}" if subject.session_id else "")
            completed += 1

            try:
                result_label, outputs, error = future.result()
                if error:
                    errors[result_label] = error
                    LOGGER.error("[%d/%d] %s failed: %s", completed, total, result_label, error)
                else:
                    results[result_label] = outputs
                    LOGGER.info(
                        "[%d/%d] %s completed: %d outputs",
                        completed,
                        total,
                        result_label,
                        len(outputs),
                    )
            except Exception as e:
                errors[label] = str(e)
                LOGGER.exception("[%d/%d] %s failed with exception", completed, total, label)

    if errors:
        LOGGER.warning("Failed subjects: %s", list(errors.keys()))

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run CAT12 parcellations from a CSV file with parallel processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  CAT12_ROOT          Root directory of CAT12 derivatives (required if not using --root)
  CAT12_OUTPUT_DIR    Output directory for parcellation results
  CAT12_ATLAS_PATHS   Comma-separated list of atlas NIfTI paths
  CAT12_ATLAS_NAMES   Comma-separated list of atlas names
  CAT12_ATLAS_LUTS    Comma-separated list of atlas LUT paths
  CAT12_ATLAS_SPACE   Space for all atlases (default: MNI152NLin2009cAsym)
  CAT12_MASK          Path to brain mask
  CAT12_WORKERS       Number of parallel workers
  CAT12_LOG_LEVEL     Logging level (DEBUG, INFO, WARNING, ERROR)

CSV File Format:
  The CSV file must have a 'subject_code' column and optionally a 'session_id' column.

Example CSV:
  subject_code,session_id
  01,baseline
  01,followup
  02,baseline

Example .env file:
  CAT12_ROOT=/path/to/cat12/derivatives
  CAT12_OUTPUT_DIR=/path/to/output
  CAT12_ATLAS_PATHS=/path/to/atlas1.nii.gz,/path/to/atlas2.nii.gz
  CAT12_ATLAS_NAMES=Schaefer400,AAL
  CAT12_ATLAS_LUTS=/path/to/atlas1.tsv,/path/to/atlas2.tsv
  CAT12_WORKERS=4
""",
    )

    parser.add_argument(
        "csv",
        type=Path,
        help="Path to CSV file with 'subject_code' and optional 'session_id' columns",
    )

    parser.add_argument(
        "--root",
        type=Path,
        help=f"CAT12 derivatives root directory (overrides {ENV_CAT12_ROOT})",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help=f"Output directory (overrides {ENV_OUTPUT_DIR})",
    )

    parser.add_argument(
        "--atlas",
        type=Path,
        action="append",
        dest="atlases",
        help="Path to atlas NIfTI file (can be specified multiple times)",
    )

    parser.add_argument(
        "--atlas-name",
        type=str,
        action="append",
        dest="atlas_names",
        help="Name for the atlas (in same order as --atlas)",
    )

    parser.add_argument(
        "--atlas-lut",
        type=Path,
        action="append",
        dest="atlas_luts",
        help="Path to atlas LUT file (in same order as --atlas)",
    )

    parser.add_argument(
        "--space",
        type=str,
        default=None,
        help="Atlas space (default: MNI152NLin2009cAsym)",
    )

    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=None,
        help=f"Number of parallel workers (overrides {ENV_WORKERS}, default: CPU count)",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing output files",
    )

    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file (default: .env in current directory)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually processing",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI execution."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Load .env file
    if args.env_file:
        try:
            from dotenv import load_dotenv

            load_dotenv(args.env_file)
        except ImportError:
            print(
                "Warning: python-dotenv not installed, cannot load .env file",
                file=sys.stderr,
            )
    else:
        _load_dotenv()

    # Set up logging
    log_level_str = args.log_level or os.getenv(ENV_LOG_LEVEL, "INFO")
    log_level = _parse_log_level(log_level_str)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Build configuration
    try:
        config = config_from_env()
    except ValueError:
        # CAT12_ROOT not set, check CLI args
        if not args.root:
            LOGGER.error("CAT12_ROOT environment variable or --root argument is required")
            return 1
        config = Cat12Config(
            input_root=args.root,
            output_dir=args.output_dir or args.root / "parcellations",
            atlases=[],
        )

    # Override with CLI arguments
    if args.root:
        config = Cat12Config(
            input_root=args.root.expanduser().resolve(),
            output_dir=config.output_dir,
            atlases=config.atlases,
            subjects=config.subjects,
            sessions=config.sessions,
            mask=config.mask,
            force=config.force,
            log_level=config.log_level,
        )

    if args.output_dir:
        config = Cat12Config(
            input_root=config.input_root,
            output_dir=args.output_dir.expanduser().resolve(),
            atlases=config.atlases,
            subjects=config.subjects,
            sessions=config.sessions,
            mask=config.mask,
            force=config.force,
            log_level=config.log_level,
        )

    if args.force:
        config = Cat12Config(
            input_root=config.input_root,
            output_dir=config.output_dir,
            atlases=config.atlases,
            subjects=config.subjects,
            sessions=config.sessions,
            mask=config.mask,
            force=True,
            log_level=config.log_level,
        )

    # Build atlases from CLI arguments
    if args.atlases:
        space = args.space or os.getenv(ENV_ATLAS_SPACE, "MNI152NLin2009cAsym")
        cli_atlases = []
        for i, atlas_path in enumerate(args.atlases):
            name = args.atlas_names[i] if args.atlas_names and i < len(args.atlas_names) else atlas_path.stem
            lut = args.atlas_luts[i] if args.atlas_luts and i < len(args.atlas_luts) else None
            cli_atlases.append(
                AtlasDefinition(
                    name=name,
                    nifti_path=atlas_path.expanduser().resolve(),
                    lut=lut.expanduser().resolve() if lut else None,
                    space=space,
                )
            )
        config = Cat12Config(
            input_root=config.input_root,
            output_dir=config.output_dir,
            atlases=cli_atlases,
            subjects=config.subjects,
            sessions=config.sessions,
            mask=config.mask,
            force=config.force,
            log_level=config.log_level,
        )

    # Validate configuration
    if not config.atlases:
        LOGGER.error("No atlases configured. Use --atlas or set CAT12_ATLAS_PATHS environment variable.")
        return 1

    # Load subjects from CSV
    try:
        subjects = load_subjects_from_csv(args.csv)
    except Exception as e:
        LOGGER.error("Failed to load CSV file: %s", e)
        return 1

    if not subjects:
        LOGGER.error("No subjects found in CSV file")
        return 1

    LOGGER.info("Loaded %d subject/session pairs from %s", len(subjects), args.csv)
    LOGGER.info("CAT12 root: %s", config.input_root)
    LOGGER.info("Output directory: %s", config.output_dir)
    LOGGER.info("Atlases: %s", [a.name for a in config.atlases])

    # Dry run mode
    if args.dry_run:
        print("\nDry run - would process:")
        for subj in subjects:
            label = f"sub-{subj.subject_code}" + (f"_ses-{subj.session_id}" if subj.session_id else "")
            print(f"  {label}")
        print(f"\nTotal: {len(subjects)} subject/session pairs")
        return 0

    # Run parcellations
    max_workers = args.workers
    results = run_parcellations_parallel(subjects, config, max_workers=max_workers)

    # Summary
    total_outputs = sum(len(outputs) for outputs in results.values())
    LOGGER.info("Completed: %d subjects, %d total output files", len(results), total_outputs)

    return 0


if __name__ == "__main__":
    sys.exit(main())

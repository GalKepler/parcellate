"""Unified CLI entry point for parcellate.

BIDS App usage (recommended)::

    parcellate <bids_dir> <output_dir> participant \\
        --pipeline {cat12,qsirecon} \\
        [--participant-label LABEL [LABEL ...]] \\
        [--session-label ID [ID ...]] \\
        [--config CONFIG.toml] \\
        [--mask {gm,wm,brain}] \\
        [--stat-tier {core,extended,diagnostic,all}] \\
        [--force] [--n-jobs N] [--n-procs N]

Legacy subcommand usage (deprecated)::

    parcellate cat12 config.toml
    parcellate qsirecon config.toml
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

#: Subcommand names that trigger legacy routing.
_LEGACY_SUBCOMMANDS = frozenset({"cat12", "qsirecon"})


def _build_bids_parser() -> argparse.ArgumentParser:
    """Build the BIDS App-style argument parser."""
    parser = argparse.ArgumentParser(
        prog="parcellate",
        description=(
            "Extract regional statistics from scalar neuroimaging maps. "
            "Follows the BIDS App positional-argument convention: "
            "parcellate <bids_dir> <output_dir> <analysis_level> --pipeline PIPELINE."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- BIDS App required positional arguments ---
    parser.add_argument(
        "bids_dir",
        type=Path,
        help="Root directory of the preprocessing derivatives (e.g. CAT12 or QSIRecon output).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory where parcellation results will be written.",
    )
    parser.add_argument(
        "analysis_level",
        choices=["participant"],
        help="Level of analysis. Only 'participant' is supported at this time.",
    )

    # --- Pipeline selection (required) ---
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=["cat12", "qsirecon"],
        help="Preprocessing pipeline that produced the input data.",
    )

    # --- BIDS App standard optional arguments ---
    parser.add_argument(
        "--participant-label",
        nargs="+",
        dest="participant_label",
        metavar="LABEL",
        help=(
            "One or more participant labels to process (without the 'sub-' prefix). "
            "Processes all participants if not specified."
        ),
    )
    parser.add_argument(
        "--session-label",
        nargs="+",
        dest="session_label",
        metavar="ID",
        help=(
            "One or more session labels to process (without the 'ses-' prefix). "
            "Processes all sessions if not specified."
        ),
    )

    # --- Atlas configuration ---
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a TOML configuration file (atlas definitions, pipeline defaults).",
    )
    parser.add_argument(
        "--atlas-config",
        type=Path,
        nargs="+",
        dest="atlas_config",
        help="Path to one or more TOML files defining atlases (overrides [[atlases]] in --config).",
    )

    # --- Processing options ---
    parser.add_argument(
        "--mask",
        help="Mask to apply during parcellation. Use 'gm', 'wm', or 'brain' for built-in MNI152 masks, or supply a path to a custom mask image.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.0,
        dest="mask_threshold",
        help=(
            "Threshold for the mask image. Voxels with mask values strictly greater than this "
            "value are included. Default: 0.0 (any non-zero voxel passes). "
            "Use values in [0, 1] for probability maps, e.g. 0.5 for >50%% probability."
        ),
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing parcellation outputs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        dest="log_level",
        help="Logging verbosity. Default: INFO.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        dest="n_jobs",
        help="Number of parallel jobs for within-subject parcellation. Default: 1.",
    )
    parser.add_argument(
        "--n-procs",
        type=int,
        default=None,
        dest="n_procs",
        help="Number of parallel processes for across-subject parcellation. Default: 1.",
    )

    return parser


def _to_pipeline_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Translate BIDS App argument names to the format expected by pipeline load_config.

    The pipeline-specific ``load_config`` functions expect argparse Namespaces
    with ``input_root``, ``output_dir``, ``subjects``, ``sessions``, etc.
    This function adapts the BIDS App names to that convention.
    """
    return argparse.Namespace(
        input_root=args.bids_dir,
        output_dir=args.output_dir,
        subjects=args.participant_label,
        sessions=args.session_label,
        mask=args.mask,
        mask_threshold=args.mask_threshold,
        force=args.force,
        log_level=args.log_level,
        n_jobs=args.n_jobs,
        n_procs=args.n_procs,
        config=args.config,
        atlas_config=args.atlas_config,
        stat_tier=args.stat_tier,
    )


def _run_bids_app(args: argparse.Namespace) -> int:
    """Dispatch a BIDS App request to the appropriate pipeline."""
    pipeline_args = _to_pipeline_namespace(args)

    if args.pipeline == "cat12":
        from parcellate.interfaces.cat12.cat12 import load_config, run_parcellations

        config = load_config(pipeline_args)
        run_parcellations(config)

    elif args.pipeline == "qsirecon":
        from parcellate.interfaces.qsirecon.qsirecon import load_config, run_parcellations

        config = load_config(pipeline_args)
        run_parcellations(config)

    return 0


def _run_legacy(command: str, remaining_argv: list[str]) -> int:
    """Run a legacy subcommand with a deprecation warning."""
    warnings.warn(
        f"'parcellate {command}' is deprecated. "
        "Use the BIDS App interface instead: "
        f"'parcellate <bids_dir> <output_dir> participant --pipeline {command}'. "
        "Legacy subcommand routing will be removed in a future major release.",
        DeprecationWarning,
        stacklevel=3,
    )

    if command == "cat12":
        from parcellate.interfaces.cat12.cat12 import main as cat12_main

        return cat12_main(remaining_argv)

    if command == "qsirecon":
        from parcellate.interfaces.qsirecon.qsirecon import main as qsirecon_main

        return qsirecon_main(remaining_argv)

    return 1


def main(argv: list[str] | None = None) -> int:
    """Unified entry point for the parcellate CLI.

    Supports two calling conventions:

    1. **BIDS App** (recommended): positional ``bids_dir``, ``output_dir``,
       ``analysis_level`` followed by ``--pipeline``.
    2. **Legacy subcommand** (deprecated): ``parcellate cat12 ...`` or
       ``parcellate qsirecon ...``.

    The legacy form is detected by peeking at the first argument.
    """
    argv = list(argv) if argv is not None else sys.argv[1:]

    # --- No arguments: print help and exit cleanly ---
    if not argv:
        _build_bids_parser().print_help()
        return 1

    # --- Legacy routing: detect old-style subcommands ---
    if argv[0] in _LEGACY_SUBCOMMANDS:
        return _run_legacy(command=argv[0], remaining_argv=argv[1:])

    # --- BIDS App parsing ---
    parser = _build_bids_parser()
    args = parser.parse_args(argv)

    try:
        return _run_bids_app(args)
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Parcellation workflow failed")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Shared orchestration utilities for parcellation pipeline interfaces.

This module consolidates the CLI argument handling, TOML config loading,
output writing, and parallel execution logic that was previously duplicated
across each pipeline interface (CAT12, QSIRecon, etc.).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TypeVar

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older environments
    import tomli as tomllib  # type: ignore[import]

from parcellate.interfaces.models import (
    AtlasDefinition,
    ParcellationConfig,
    ParcellationOutput,
    ReconInput,
)
from parcellate.interfaces.utils import (
    _as_list,
    _parse_log_level,
    _parse_mask,
    parse_atlases,
    write_parcellation_sidecar,
)

LOGGER = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT", bound=ParcellationConfig)


def add_cli_args(
    parser: argparse.ArgumentParser,
    input_help: str = "Root directory of preprocessing derivatives.",
) -> None:
    """Add common CLI arguments shared across all pipeline interfaces.

    Parameters
    ----------
    parser
        The argument parser to add arguments to.
    input_help
        Help text for the ``--input-root`` argument; defaults to a generic
        description but should be customised for each pipeline.
    """
    parser.add_argument(
        "--input-root",
        type=Path,
        help=input_help,
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
        help="Mask to apply during parcellation. Use 'gm', 'wm', or 'brain' for built-in MNI152 masks, or supply a path to a custom mask image.",
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


def load_config_base(
    args: argparse.Namespace,
    config_cls: type[ConfigT],
    default_atlas_space: str | None = None,
    **extra_kwargs: object,
) -> ConfigT:
    """Parse a TOML configuration file and override with CLI arguments.

    This is the shared implementation for all pipeline ``load_config`` functions.
    Pipeline-specific wrappers supply ``config_cls`` and ``default_atlas_space``.

    Parameters
    ----------
    args
        Parsed CLI arguments (from :func:`argparse.ArgumentParser.parse_args`).
    config_cls
        The concrete :class:`~parcellate.interfaces.models.ParcellationConfig`
        subclass to instantiate (e.g., ``Cat12Config``, ``QSIReconConfig``).
    default_atlas_space
        Space identifier to assign to atlases that do not specify one. Pass
        ``None`` to leave the space unset for pipelines where atlas space must
        always come from the atlas definition (e.g. QSIRecon).
    **extra_kwargs
        Additional keyword arguments forwarded to ``config_cls.__init__``, allowing
        pipeline-specific fields to be set (e.g. a pipeline-specific default mask).

    Returns
    -------
    ConfigT
        A fully initialised pipeline configuration object.
    """
    data: dict[str, object] = {}
    if args.config:
        with args.config.open("rb") as f:
            data = tomllib.load(f)

    input_root_str = args.input_root or data.get("input_root", ".")
    input_root = Path(str(input_root_str)).expanduser().resolve()
    output_dir_str = args.output_dir or data.get("output_dir", input_root / "parcellations")
    output_dir = Path(str(output_dir_str)).expanduser().resolve()
    subjects = args.subjects or _as_list(data.get("subjects"))
    sessions = args.sessions or _as_list(data.get("sessions"))

    atlas_configs: list[dict[str, object]] = []
    if args.atlas_config:
        for atlas_path in args.atlas_config:
            with atlas_path.open("rb") as f:
                atlas_configs.append(tomllib.load(f))
    else:
        atlas_configs = data.get("atlases", [])  # type: ignore[assignment]
    atlases = parse_atlases(atlas_configs, default_space=default_atlas_space)

    mask_value = args.mask or data.get("mask")
    mask = _parse_mask(mask_value)
    mask_threshold = float(getattr(args, "mask_threshold", None) or data.get("mask_threshold", 0.0))
    force = args.force or bool(data.get("force", False))
    log_level = _parse_log_level(args.log_level or data.get("log_level"))
    n_jobs = args.n_jobs or int(data.get("n_jobs", 1))
    n_procs = args.n_procs or int(data.get("n_procs", 1))

    return config_cls(
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
        **extra_kwargs,
    )


def write_output(
    result: ParcellationOutput,
    destination: Path,
    config: ParcellationConfig,
    build_output_path_fn: Callable[..., Path],
) -> Path:
    """Write a parcellation result (TSV + JSON sidecar) to disk.

    Parameters
    ----------
    result
        The parcellation output to write.
    destination
        Root output directory.
    config
        Parcellation configuration (used for mask, thresholds, etc.).
    build_output_path_fn
        A callable with the same signature as a pipeline's ``_build_output_path``
        that constructs the output file path from context, atlas, and scalar map.

    Returns
    -------
    Path
        Path to the written TSV file.
    """
    out_path = build_output_path_fn(
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


def run_parallel_workflow(
    config: ParcellationConfig,
    recon_inputs: list[ReconInput],
    run_recon_fn: Callable[[ReconInput, ParcellationConfig], list[Path]],
    pipeline_name: str = "parcellate",
) -> list[Path]:
    """Execute a parcellation workflow over all subjects, with optional parallelism.

    Respects ``config.n_procs``: when > 1, subjects are processed in a
    :class:`~concurrent.futures.ProcessPoolExecutor`; otherwise they are
    processed sequentially.

    Parameters
    ----------
    config
        Parsed pipeline configuration.
    recon_inputs
        List of per-subject/session reconstruction inputs.
    run_recon_fn
        Callable that runs the parcellation for a single recon input and returns
        a list of output paths.  Signature: ``(recon, config) -> list[Path]``.
    pipeline_name
        Human-readable name used in log messages (e.g. ``"CAT12"``).

    Returns
    -------
    list[Path]
        All output paths produced across all subjects.
    """
    outputs: list[Path] = []
    total = len(recon_inputs)

    if config.n_procs > 1:
        LOGGER.info("Running %s parcellation across subjects with %d processes", pipeline_name, config.n_procs)
        with ProcessPoolExecutor(max_workers=config.n_procs) as executor:
            future_to_recon = {executor.submit(run_recon_fn, recon, config): recon for recon in recon_inputs}
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
                result = run_recon_fn(recon, config)
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


def build_pending_plan(
    recon: ReconInput,
    config: ParcellationConfig,
    plan: dict[AtlasDefinition, list],
    build_output_path_fn: Callable[..., Path],
    *,
    catch_value_error: bool = False,
) -> tuple[dict[AtlasDefinition, list], list[Path]]:
    """Identify which parcellation outputs need to be (re)computed.

    Iterates over the planned atlas/scalar-map pairs and checks whether output
    files already exist.  When ``config.force`` is ``False`` and a file exists,
    the path is added to ``reused_outputs``; otherwise the scalar map is kept in
    ``pending_plan`` for actual processing.

    Parameters
    ----------
    recon
        The reconstruction input for a single subject/session.
    config
        Parcellation configuration (used for ``output_dir``, ``force``, and
        masking parameters).
    plan
        Mapping of atlases to scalar maps from the workflow planner.
    build_output_path_fn
        Callable that constructs the output path for a given atlas/scalar-map
        combination.
    catch_value_error
        When ``True``, ``ValueError`` exceptions raised by ``build_output_path_fn``
        are silently caught with a warning log, and the scalar map is skipped.
        Set this for pipelines where some combinations may not have all required
        BIDS entities (e.g. QSIRecon ``param`` field).

    Returns
    -------
    tuple[dict, list[Path]]
        ``(pending_plan, reused_outputs)`` where ``pending_plan`` maps each atlas
        to the scalar maps that still need processing, and ``reused_outputs`` is
        the list of already-existing paths that will be returned as-is.
    """
    pending_plan: dict[AtlasDefinition, list] = {}
    reused_outputs: list[Path] = []

    for atlas, scalar_maps in plan.items():
        remaining = []
        for scalar_map in scalar_maps:
            try:
                out_path = build_output_path_fn(
                    context=recon.context,
                    atlas=atlas,
                    scalar_map=scalar_map,
                    destination=config.output_dir,
                    mask=config.mask,
                    mask_threshold=config.mask_threshold,
                    atlas_threshold=atlas.atlas_threshold,
                )
            except ValueError:
                if catch_value_error:
                    LOGGER.warning(
                        "Could not build output path for %s, %s, %s",
                        recon.context,
                        atlas,
                        scalar_map,
                    )
                    continue
                raise
            if not config.force and out_path.exists():
                LOGGER.info("Reusing existing parcellation output at %s", out_path)
                reused_outputs.append(out_path)
            else:
                remaining.append(scalar_map)
        if remaining:
            pending_plan[atlas] = remaining

    return pending_plan, reused_outputs

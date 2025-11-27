"""High-level orchestration for parcellating QSIRecon outputs.

This module provides a small CLI that reads a TOML configuration file,
loads recon inputs, plans a parcellation workflow, runs it, and writes the
outputs into a BIDS-derivative-style directory mirroring the structure used
by QSIRecon.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older environments
    import tomli as tomllib  # type: ignore

from parcellate.interfaces.qsirecon.loader import load_qsirecon_inputs
from parcellate.interfaces.qsirecon.planner import plan_qsirecon_parcellation_workflow
from parcellate.interfaces.qsirecon.runner import run_qsirecon_parcellation_workflow
from parcellate.interfaces.qsirecon.models import ParcellationOutput


LOGGER = logging.getLogger(__name__)


@dataclass
class QSIReconConfig:
    """Configuration parsed from TOML input."""

    input_root: Path
    output_dir: Path
    subjects: list[str] | None
    sessions: list[str] | None
    mask: Path | None
    log_level: int = logging.INFO


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
    log_level = _parse_log_level(data.get("log_level"))

    return QSIReconConfig(
        input_root=input_root,
        output_dir=output_dir,
        subjects=subjects,
        sessions=sessions,
        mask=mask,
        log_level=log_level,
    )


def _as_list(value: Iterable[str] | str | None) -> list[str] | None:
    """Normalize configuration values into a list of strings."""

    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _write_output(result: ParcellationOutput, destination: Path) -> Path:
    """Write a parcellation output to disk using a QSIRecon-like layout."""

    workflow = result.scalar_map.recon_workflow or "parcellate"
    base = destination / f"qsirecon-{workflow}"

    subject_dir = base / f"sub-{result.context.subject_id}"
    if result.context.session_id:
        subject_dir = subject_dir / f"ses-{result.context.session_id}"

    # QSIRecon organizes diffusion derivatives under ``dwi``
    output_dir = subject_dir / "dwi"
    output_dir.mkdir(parents=True, exist_ok=True)

    entities: list[str] = [result.context.label]
    space = result.atlas.space or result.scalar_map.space
    if space:
        entities.append(f"space-{space}")
    if result.atlas.resolution:
        entities.append(f"res-{result.atlas.resolution}")
    entities.append(f"atlas-{result.atlas.name}")
    if result.scalar_map.model:
        entities.append(f"model-{result.scalar_map.model}")
    if result.scalar_map.origin:
        entities.append(f"desc-{result.scalar_map.origin}")
    entities.append(f"scalar-{result.scalar_map.name}")

    filename = "_".join(entities + ["parcellation"]) + ".tsv"
    out_path = output_dir / filename
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
        if config.mask:
            recon.mask = config.mask
        plan = plan_qsirecon_parcellation_workflow(recon)
        jobs = run_qsirecon_parcellation_workflow(recon=recon, plan=plan)
        for result in jobs:
            outputs.append(_write_output(result, destination=config.output_dir))
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

"""IO utilities for discovering QSIRecon outputs."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from parcellate.interfaces.qsirecon.models import (
    AtlasDefinition,
    ReconInput,
    ScalarMapDefinition,
    SubjectContext,
)
from parcellate.interfaces.utils import parse_bids_entities

logger = logging.getLogger(__name__)


def _process_subject_session(
    root: Path,
    subject_id: str,
    session_id: str | None,
    atlas_definitions: list[AtlasDefinition],
) -> ReconInput | None:
    """Process a single subject/session to discover inputs."""
    try:
        context = SubjectContext(subject_id=subject_id, session_id=session_id)
        scalar_maps = discover_scalar_maps(root=root, subject=subject_id, session=session_id)
        if not scalar_maps:
            logger.debug(f"No scalar maps found for sub-{subject_id} ses-{session_id}")
            return None
        return ReconInput(
            context=context,
            scalar_maps=scalar_maps,
            atlases=atlas_definitions,
            transforms=(),
        )
    except Exception:
        logger.exception(f"Error processing sub-{subject_id} ses-{session_id}")
        return None


def load_qsirecon_inputs(
    root: Path,
    subjects: Iterable[str] | None = None,
    sessions: Iterable[str] | None = None,
    atlases: Iterable[AtlasDefinition] | None = None,
    max_workers: int | None = None,
) -> list[ReconInput]:
    """Discover scalar maps and atlases for subjects/sessions in a QSIRecon derivative."""
    root = Path(root)
    subj_list = list(subjects) if subjects else _discover_subjects(root)
    atlas_definitions = list(atlases) if atlases else discover_atlases(root=root)
    logger.info(
        f"Discovered {len(subj_list)} subjects and {len(atlas_definitions)} atlases in QSIRecon derivatives. "
        f"Processing with up to {max_workers or 'unlimited'} workers."
    )

    tasks = []
    for subject_id in subj_list:
        ses_list = list(sessions) if sessions else _discover_sessions(root, subject_id)
        for session_id in ses_list:
            tasks.append((subject_id, session_id))

    recon_inputs: list[ReconInput] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_subject_session,
                root,
                subject_id,
                session_id,
                atlas_definitions,
            )
            for subject_id, session_id in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                recon_inputs.append(result)
    return recon_inputs


def discover_scalar_maps(root: Path, subject: str, session: str | None) -> list[ScalarMapDefinition]:
    """Return scalar map definitions."""
    root = Path(root)
    scalar_maps: list[ScalarMapDefinition] = []

    for workflow_dir in sorted(root.glob("qsirecon-*")):
        search_dir = workflow_dir / f"sub-{subject}"
        if session:
            search_dir = search_dir / f"ses-{session}"
        if not search_dir.exists():
            continue
        for nii_path in sorted(search_dir.rglob("*_dwimap.nii*")):
            entities = parse_bids_entities(nii_path.name)
            scalar_maps.append(
                ScalarMapDefinition(
                    name=_scalar_name(entities, nii_path.name, nii_path),
                    nifti_path=nii_path,
                    param=_parameter_name(entities, nii_path.name),
                    desc=entities.get("desc"),
                    model=entities.get("model"),
                    origin=entities.get("Description"),
                    space=entities.get("space"),
                    recon_workflow=_workflow_name(root, nii_path),
                )
            )

    return scalar_maps


def discover_atlases(
    root: Path,
    space: str = "MNI152NLin2009cAsym",
    allow_fallback: bool = True,
    subject: str | None = None,
    session: str | None = None,
    **kwargs: str,
) -> list[AtlasDefinition]:
    """Return atlas definitions."""
    root = Path(root)
    atlas_files = _find_dseg_files(root, space=space, subject=subject, session=session)
    if not atlas_files and allow_fallback:
        atlas_files = _find_dseg_files(root, space=None, subject=subject, session=session)

    atlases: list[AtlasDefinition] = []
    for atlas_path in atlas_files:
        entities = parse_bids_entities(atlas_path.name)
        name = (
            entities.get("segmentation")
            or entities.get("atlas")
            or entities.get("desc")
            or atlas_path.name.split(".nii")[0]
        )
        resolution = entities.get("resolution")
        atlas_space = entities.get("space") or space
        lut_path = _find_lut_file(atlas_path, name, root)
        atlases.append(
            AtlasDefinition(
                name=name,
                nifti_path=atlas_path,
                lut=lut_path,
                resolution=resolution,
                space=atlas_space,
            )
        )
    return atlases


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _discover_subjects(root: Path) -> list[str]:
    """Discover subject identifiers from qsirecon workflow directories."""
    subjects: set[str] = set()
    for workflow_dir in root.glob("qsirecon-*"):
        logger.info(f"Checking workflow directory {workflow_dir}")
        for subj_dir in workflow_dir.glob("sub-*"):
            if subj_dir.is_dir():
                subjects.add(subj_dir.name.replace("sub-", "", 1))
    return sorted(subjects)


def _discover_sessions(root: Path, subject: str) -> list[str | None]:
    """Discover session identifiers for a subject across all workflow dirs."""
    sessions: set[str] = set()
    for workflow_dir in root.glob("qsirecon-*"):
        subj_dir = workflow_dir / f"sub-{subject}"
        for ses_dir in subj_dir.glob("ses-*"):
            if ses_dir.is_dir():
                sessions.add(ses_dir.name.replace("ses-", "", 1))
    return sorted(sessions) if sessions else [None]


def _find_dseg_files(
    root: Path,
    space: str | None = None,
    subject: str | None = None,
    session: str | None = None,
) -> list[Path]:
    """Find dseg NIfTI files, optionally filtering by space."""
    results: list[Path] = []
    for nii_path in sorted(root.rglob("*_dseg.nii*")):
        entities = parse_bids_entities(nii_path.name)
        if subject and entities.get("subject") != subject:
            continue
        if session and entities.get("session") != session:
            continue
        if space and entities.get("space", "").lower() != space.lower():
            continue
        results.append(nii_path)
    return results


def _find_lut_file(atlas_path: Path, atlas_name: str, root: Path) -> Path | None:
    """Search for a .tsv/.csv look-up table matching *atlas_name*.

    Searches the atlas file's directory first, then the broader tree.
    """
    for search_root in (atlas_path.parent, root):
        for ext in ("tsv", "csv"):
            for candidate in search_root.rglob(f"*{atlas_name}*.{ext}"):
                return candidate
    return None


def _parameter_name(entities: dict[str, str], filename: str) -> str:
    """Extract parameter name from parsed entities or filename."""
    param = entities.get("param")
    if not param:
        param = filename.split("param-")[-1].split("_")[0] if "param-" in filename else ""
    return param


def _scalar_name(entities: dict[str, str], filename: str, filepath: Path) -> str:
    """Construct a scalar map name from parsed entities."""
    name_parts = [
        entities.get("model"),
        _parameter_name(entities, filename),
        entities.get("desc"),
    ]
    name = "-".join(part for part in name_parts if part)
    return name or filepath.name.split(".nii")[0]


def _workflow_name(root: Path, filepath: Path) -> str:
    """Extract workflow name from a file path by finding the qsirecon-* component."""
    try:
        rel = filepath.relative_to(root)
    except ValueError:
        return filepath.parent.name
    for part in rel.parts:
        if part.startswith("qsirecon-"):
            return part.split("qsirecon-", 1)[1]
    return filepath.parent.name

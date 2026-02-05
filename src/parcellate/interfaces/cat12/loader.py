"""IO utilities for discovering CAT12 outputs."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from parcellate.interfaces.cat12.models import (
    AtlasDefinition,
    ReconInput,
    ScalarMapDefinition,
    SubjectContext,
    TissueType,
)

logger = logging.getLogger(__name__)

# CAT12 tissue type patterns
TISSUE_PATTERNS: dict[TissueType, str] = {
    TissueType.GM: "mwp1*",  # Gray matter probability maps
    TissueType.WM: "mwp2*",  # White matter probability maps
    TissueType.CT: "wct*",  # Cortical thickness maps
}

# Default space for CAT12 normalized outputs
DEFAULT_SPACE = "MNI152NLin2009cAsym"


def load_cat12_inputs(
    root: Path,
    atlases: Sequence[AtlasDefinition] | None = None,
    subjects: Iterable[str] | None = None,
    sessions: Iterable[str] | None = None,
) -> list[ReconInput]:
    """Discover scalar maps for subjects/sessions in a CAT12 derivative.

    Parameters
    ----------
    root
        Root directory of CAT12 derivatives.
    atlases
        List of atlas definitions to use for parcellation.
    subjects
        Optional list of subject identifiers to process.
    sessions
        Optional list of session identifiers to process.

    Returns
    -------
    list[ReconInput]
        List of ReconInput instances for each subject/session.
    """
    root = Path(root)
    recon_inputs: list[ReconInput] = []

    if not atlases:
        raise ValueError("At least one atlas must be configured for the CAT12 interface.")

    # Discover subjects
    subj_list = list(subjects) if subjects else _discover_subjects(root)
    if not subj_list:
        logger.warning("No subjects found in %s", root)
        return []

    for subject_id in subj_list:
        # Discover sessions for this subject
        ses_list = list(sessions) if sessions else _discover_sessions(root, subject_id)
        if not ses_list:
            ses_list = [None]

        for session_id in ses_list:
            context = SubjectContext(subject_id=subject_id, session_id=session_id)
            scalar_maps = discover_scalar_maps(root=root, subject=subject_id, session=session_id)

            if scalar_maps:
                recon_inputs.append(
                    ReconInput(
                        context=context,
                        scalar_maps=scalar_maps,
                        atlases=atlases,
                        transforms=(),
                    )
                )
            else:
                logger.debug(
                    "No scalar maps found for subject %s session %s",
                    subject_id,
                    session_id,
                )

    return recon_inputs


def _discover_subjects(root: Path) -> list[str]:
    """Discover subject identifiers from directory structure.

    Parameters
    ----------
    root
        Root directory of CAT12 derivatives.

    Returns
    -------
    list[str]
        List of subject identifiers (without 'sub-' prefix).
    """
    subjects = []
    for path in root.glob("sub-*"):
        if path.is_dir():
            subj_id = path.name.replace("sub-", "")
            subjects.append(subj_id)
    return sorted(subjects)


def _discover_sessions(root: Path, subject: str) -> list[str | None]:
    """Discover session identifiers for a subject.

    Parameters
    ----------
    root
        Root directory of CAT12 derivatives.
    subject
        Subject identifier (without 'sub-' prefix).

    Returns
    -------
    list[str | None]
        List of session identifiers (without 'ses-' prefix), or [None] if no sessions.
    """
    subject_dir = root / f"sub-{subject}"
    sessions = []
    for path in subject_dir.glob("ses-*"):
        if path.is_dir():
            ses_id = path.name.replace("ses-", "")
            sessions.append(ses_id)
    return sorted(sessions) if sessions else [None]


def _build_search_path(root: Path, subject: str, session: str | None) -> Path:
    """Build the search path for CAT12 outputs.

    Parameters
    ----------
    root
        Root directory of CAT12 derivatives.
    subject
        Subject identifier (without 'sub-' prefix).
    session
        Session identifier (without 'ses-' prefix), or None.

    Returns
    -------
    Path
        Path to search for CAT12 outputs.
    """
    search_path = root / f"sub-{subject}"
    if session:
        search_path = search_path / f"ses-{session}"
    search_path = search_path / "anat"
    return search_path


def discover_scalar_maps(
    root: Path,
    subject: str,
    session: str | None,
) -> list[ScalarMapDefinition]:
    """Discover CAT12 scalar maps for a subject/session.

    Searches for:
    - mwp1*.nii* files -> GM tissue maps
    - mwp2*.nii* files -> WM tissue maps
    - wct*.nii* files -> CT (cortical thickness) maps

    Parameters
    ----------
    root
        Root directory of CAT12 derivatives.
    subject
        Subject identifier (without 'sub-' prefix).
    session
        Session identifier (without 'ses-' prefix), or None.

    Returns
    -------
    list[ScalarMapDefinition]
        List of discovered scalar map definitions.
    """
    search_path = _build_search_path(root, subject, session)
    scalar_maps: list[ScalarMapDefinition] = []

    if not search_path.exists():
        logger.debug("Search path does not exist: %s", search_path)
        return scalar_maps

    for tissue_type, pattern in TISSUE_PATTERNS.items():
        for nii_path in search_path.glob(f"{pattern}.nii*"):
            name = _scalar_name(nii_path, tissue_type)
            scalar_maps.append(
                ScalarMapDefinition(
                    name=name,
                    nifti_path=nii_path,
                    tissue_type=tissue_type,
                    space=DEFAULT_SPACE,  # CAT12 normalized outputs are in MNI space
                    desc=_extract_desc(nii_path),
                )
            )

    return scalar_maps


def _scalar_name(nii_path: Path, tissue_type: TissueType) -> str:
    """Construct a scalar map name from a NIfTI file path.

    Parameters
    ----------
    nii_path
        Path to the NIfTI file.
    tissue_type
        The tissue type classification.

    Returns
    -------
    str
        The constructed scalar map name.
    """
    stem = nii_path.name.split(".nii")[0]
    return f"{tissue_type.value}-{stem}"


def _extract_desc(nii_path: Path) -> str | None:
    """Extract description from a NIfTI file path.

    Parameters
    ----------
    nii_path
        Path to the NIfTI file.

    Returns
    -------
    str | None
        Extracted description, or None.
    """
    stem = nii_path.name.split(".nii")[0]
    # CAT12 files don't typically have BIDS-style desc entity
    # Return the full stem as description
    return stem

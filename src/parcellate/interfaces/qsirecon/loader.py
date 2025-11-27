"""IO utilities for discovering QSIRecon outputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from bids import BIDSLayout

from parcellate.interfaces.qsirecon.models import (
    AtlasDefinition,
    ReconInput,
    ScalarMapDefinition,
    SubjectContext,
)


def load_qsirecon_inputs(
    root: Path,
    subjects: Iterable[str] | None = None,
    sessions: Iterable[str] | None = None,
) -> list[ReconInput]:
    """Discover scalar maps and atlases for subjects/sessions in a QSIRecon derivative."""

    layout = BIDSLayout(
        root,
        validate=False,
        derivatives=True,
        config=["bids", "derivatives"],
    )
    entities = layout.get_entities()
    subj_list = list(subjects) if subjects else layout.get_subjects()
    recon_inputs: list[ReconInput] = []
    atlases = discover_atlases(layout=layout)
    for subject_id in subj_list:
        if sessions:
            ses_list = list(sessions)
        elif "session" in entities:
            ses_list = layout.get_sessions(subject=subject_id) or [None]
        else:
            ses_list = [None]
        for session_id in ses_list:
            context = SubjectContext(subject_id=subject_id, session_id=session_id)
            scalar_maps = discover_scalar_maps(layout=layout, subject=subject_id, session=session_id)
            recon_inputs.append(
                ReconInput(
                    context=context,
                    scalar_maps=scalar_maps,
                    atlases=atlases,
                    mask=None,
                    transforms=(),
                )
            )
    return recon_inputs


def discover_scalar_maps(layout: BIDSLayout, subject: str, session: str | None) -> list[ScalarMapDefinition]:
    """Return scalar map definitions."""

    filters = {
        "subject": subject,
        "suffix": "dwimap",
        "extension": ["nii", "nii.gz"],
    }
    if session and "session" in layout.get_entities():
        filters["session"] = session

    files = layout.get(
        return_type="object",
        **filters,
    )

    scalar_maps: list[ScalarMapDefinition] = []

    for fobj in files:
        scalar_maps.append(
            ScalarMapDefinition(
                name=_scalar_name(fobj),
                nifti_path=Path(fobj.path),
                model=fobj.entities.get("model"),
                origin=fobj.entities.get("Description"),
                space=fobj.entities.get("space"),
                recon_workflow=_workflow_name(layout, fobj),
            )
        )

    return scalar_maps


def discover_atlases(
    layout: BIDSLayout,
    space: str = "MNI152NLin2009cAsym",
    allow_fallback: bool = True,
    subject: str | None = None,
    session: str | None = None,
    **kwargs,
) -> list[AtlasDefinition]:
    """Return atlas definitions."""

    filters = {
        "space": space,
        "suffix": ["dseg"],
        "extension": ["nii", "nii.gz"],
        "subject": subject,
        "session": session,
        **kwargs,
    }

    filters = {k: v for k, v in filters.items() if v is not None}

    atlas_files = layout.get(return_type="object", **filters)
    if not atlas_files and allow_fallback:
        # Fallback: drop space constraint if not found
        fallback = {k: v for k, v in filters.items() if k != "space"}
        atlas_files = layout.get(return_type="object", **fallback)

    atlases: list[AtlasDefinition] = []
    for fobj in atlas_files:
        name = (
            fobj.get_entities().get("segmentation")
            or fobj.get_entities().get("atlas")
            or fobj.get_entities().get("desc")
            or Path(fobj.path).stem
        )
        resolution = fobj.get_entities().get("res") or None
        space = fobj.get_entities().get("space") or filters.get("space")
        lut_entities = {"atlas": name, "extension": ["tsv", "csv"]}
        lut_files = layout.get(return_type="object", **lut_entities)

        lut_path = Path(lut_files[0].path) if lut_files else None
        atlases.append(
            AtlasDefinition(
                name=name,
                nifti_path=Path(fobj.path),
                lut=lut_path,
                resolution=resolution,
                space=space,
            )
        )
    return atlases


def _parameter_name(fobj) -> str:
    entities = fobj.get_entities()
    param = entities.get("param")
    if not param:
        fname = fobj.filename
        param_value = fname.split("param-")[-1].split("_")[0]
        param = param_value
    return param


def _scalar_name(fobj) -> str:
    entities = fobj.get_entities()
    name_parts = [entities.get("model"), _parameter_name(fobj), entities.get("desc")]
    name = "-".join(part for part in name_parts if part)
    return name or Path(fobj.path).stem


def _workflow_name(layout, fobj) -> str:
    entities = fobj.get_entities()
    workflow = entities.get("recon_workflow")
    if not workflow:
        fname = Path(fobj.path).relative_to(layout.root).parts[1]
        workflow_value = fname.split("qsirecon-")[-1]
        workflow = workflow_value
    return workflow

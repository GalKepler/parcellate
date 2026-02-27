"""Run parcellation workflows.

This module provides functions for running parcellation workflows.
"""

import logging
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union

from parcellate.interfaces.models import (
    AtlasDefinition,
    ParcellationConfig,
    ParcellationOutput,
    ReconInput,
)
from parcellate.parcellation.volume import VolumetricParcellator

logger = logging.getLogger(__name__)


class ScalarMapSpaceMismatchError(ValueError):
    """Raised when scalar maps have inconsistent spaces."""

    def __init__(self, spaces: set[Union[str, None]]):
        """Initialize the error."""
        super().__init__(f"Scalar maps have inconsistent spaces: {spaces}")


def _validate_scalar_map_spaces(scalar_maps: list) -> None:
    """Validate that all scalar maps share the same space.

    Parameters
    ----------
    scalar_maps : list
        List of scalar map definitions to validate.

    Raises
    ------
    ScalarMapSpaceMismatchError
        If scalar maps have different spaces.
    """
    if not scalar_maps:
        return
    spaces = {sm.space for sm in scalar_maps}
    if len(spaces) > 1:
        raise ScalarMapSpaceMismatchError(spaces)


def _parcellate_scalar_map(
    recon: ReconInput,
    atlas: AtlasDefinition,
    scalar_map,
    parcellator: VolumetricParcellator,
) -> Optional[ParcellationOutput]:
    """Parcellate a single scalar map."""
    logger.debug("Parcellating %s with atlas %s", scalar_map.name, atlas.name)
    try:
        stats_table = parcellator.transform(scalar_img=scalar_map.nifti_path)
        po = ParcellationOutput(
            context=recon.context,
            atlas=atlas,
            scalar_map=scalar_map,
            stats_table=stats_table,
        )
        logger.info(
            "Successfully parcellated %s with atlas %s",
            scalar_map.name,
            atlas.name,
        )
        return po  # noqa: TRY300
    except Exception:  # Broad catch intentional: defensive processing in batch pipelines
        logger.exception(
            "Failed to parcellate %s with atlas %s",
            scalar_map.name,
            atlas.name,
        )
        return None


def run_parcellation_workflow(
    recon: ReconInput,
    plan: Mapping[AtlasDefinition, list],
    config: ParcellationConfig,
) -> list[ParcellationOutput]:
    """Run parcellation workflow for a given input.

    Parameters
    ----------
    recon
        ReconInput instance describing the inputs for a subject/session.
    plan
        Mapping of atlas definitions to lists of scalar maps to parcellate.
    config
        Configuration for the parcellation workflow.

    Returns
    -------
    list[ParcellationOutput]
        List of parcellation outputs generated.
    """
    logger.info("Running parcellation workflow for %s", recon)
    jobs: list[ParcellationOutput] = []

    num_jobs = sum(len(scalar_maps) for scalar_maps in plan.values())
    logger.info("Found %d parcellation jobs to run.", num_jobs)

    # Phase 1 (sequential): validate spaces, initialise and fit one parcellator per atlas
    parcellators: dict[AtlasDefinition, tuple[VolumetricParcellator, list]] = {}
    for atlas, scalar_maps in plan.items():
        if not scalar_maps:
            logger.debug("No scalar maps to parcellate for atlas %s", atlas.name)
            continue

        try:
            _validate_scalar_map_spaces(scalar_maps)
        except ScalarMapSpaceMismatchError as e:
            logger.warning("Skipping atlas %s due to space mismatch: %s", atlas.name, e)
            continue

        logger.info("Initializing parcellator for atlas %s", atlas.name)
        try:
            vp = VolumetricParcellator(
                atlas_img=atlas.nifti_path,
                lut=atlas.lut,
                mask=config.mask,
                mask_threshold=config.mask_threshold,
                atlas_threshold=atlas.atlas_threshold,
                background_label=config.background_label,
                resampling_target=config.resampling_target,
                stat_tier=config.stat_tier,
            )
            vp.fit(scalar_img=scalar_maps[0].nifti_path)
            parcellators[atlas] = (vp, scalar_maps)
        except Exception:
            logger.exception("Failed to initialize parcellator for atlas %s", atlas.name)
            continue

    # Phase 2 (parallel): submit all scalar-map jobs across all atlases in one pool
    with ThreadPoolExecutor(max_workers=config.n_jobs) as executor:
        future_to_sm = {
            executor.submit(_parcellate_scalar_map, recon, atlas, sm, vp): (atlas, sm)
            for atlas, (vp, scalar_maps) in parcellators.items()
            for sm in scalar_maps
        }
        for future in as_completed(future_to_sm):
            result = future.result()
            if result:
                jobs.append(result)

    logger.info("Finished parcellation workflow for %s", recon)
    return jobs

"""Plan parcellation workflows.

This module provides functions for planning parcellation workflows.
"""

import logging
from collections.abc import Mapping

from parcellate.interfaces.models import AtlasDefinition, ReconInput

logger = logging.getLogger(__name__)


def _space_match(atlas: AtlasDefinition, scalar_map) -> bool:
    """Return whether atlas and scalar map share the same space."""
    return bool(atlas.space and scalar_map.space and atlas.space.lower() == scalar_map.space.lower())


def plan_parcellation_workflow(
    recon_input: ReconInput,
) -> Mapping[AtlasDefinition, list]:
    """Plan parcellation workflow for a given recon input.

    Parameters
    ----------
    recon_input
        ReconInput instance describing the inputs for a subject/session.

    Returns
    -------
    Mapping[AtlasDefinition, list[ScalarMapDefinition]]
        Mapping of atlases to scalar maps to be parcellated.
    """
    logger.info("Planning parcellation workflow for %s", recon_input)
    plan: dict[AtlasDefinition, list] = {}
    for atlas in recon_input.atlases:
        logger.debug("Considering atlas %s", atlas.name)
        plan[atlas] = [scalar_map for scalar_map in recon_input.scalar_maps if _space_match(atlas, scalar_map)]
        logger.debug("Matched %d scalar maps to atlas %s", len(plan[atlas]), atlas.name)
    logger.info("Planned parcellation workflow: %s", plan)
    return plan

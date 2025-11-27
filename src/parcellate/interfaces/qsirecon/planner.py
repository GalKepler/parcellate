"""
Plan QSIRECON parcellation workflows.
"""

from collections.abc import Mapping

from parcellate.interfaces.qsirecon.models import AtlasDefinition, ScalarMapDefinition


def _space_match(atlas: AtlasDefinition, scalar_map: ScalarMapDefinition) -> bool:
    """Return whether atlas and scalar map share the same space."""

    return bool(atlas.space and scalar_map.space and atlas.space.lower() == scalar_map.space.lower())


def plan_qsirecon_parcellation_workflow(recon_input) -> Mapping[AtlasDefinition : list[ScalarMapDefinition]]:
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

    plan: dict[AtlasDefinition, list[ScalarMapDefinition]] = {}
    for atlas in recon_input.atlases:
        plan[atlas] = [scalar_map for scalar_map in recon_input.scalar_maps if _space_match(atlas, scalar_map)]
    return plan

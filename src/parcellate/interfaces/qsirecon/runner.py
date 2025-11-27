from collections.abc import Mapping

from parcellate.interfaces.qsirecon.models import AtlasDefinition, ParcellationOutput, ReconInput, ScalarMapDefinition
from parcellate.parcellation.volume import VolumetricParcellator


def run_qsirecon_parcellation_workflow(
    recon: ReconInput,
    plan: Mapping[AtlasDefinition, list[ScalarMapDefinition]],
) -> list[ParcellationOutput]:
    """Run parcellation workflow for a given recon input.

    Parameters
    ----------
    recon
        ReconInput instance describing the inputs for a subject/session.

    Returns
    -------
    list[ParcellationOutput]
        List of parcellation outputs generated.
    """
    jobs: list[ParcellationOutput] = []

    for atlas, scalar_maps in plan.items():
        vp = VolumetricParcellator(
            atlas_img=atlas.nifti_path,
            lut=atlas.lut,
        )
        vp.fit(scalar_img=scalar_maps[0].nifti_path)
        for scalar_map in scalar_maps:
            stats_table = vp.transform(scalar_img=scalar_map.nifti_path)
            po = ParcellationOutput(
                context=recon.context,
                atlas=atlas,
                scalar_map=scalar_map,
                stats_table=stats_table,
            )
            jobs.append(po)
    return jobs

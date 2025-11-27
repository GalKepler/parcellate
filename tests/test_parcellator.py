import nibabel as nib
import numpy as np
import pytest

from parcellate import Parcellator


def _atlas() -> nib.Nifti1Image:
    data = np.array(
        [
            [[0, 1], [1, 2]],
            [[0, 2], [2, 2]],
        ],
        dtype=np.int16,
    )
    return nib.Nifti1Image(data, np.eye(4))


def test_parcellate_returns_expected_statistics() -> None:
    atlas_img = _atlas()
    map_data = np.array(
        [
            [[0.0, 2.0], [4.0, 6.0]],
            [[0.0, 8.0], [10.0, 12.0]],
        ],
    )
    map_img = nib.Nifti1Image(map_data, atlas_img.affine)

    parcellator = Parcellator(atlas_img, labels={1: "hippocampus", 2: "amygdala"})
    df = parcellator.parcellate(map_img)

    atlas_data = atlas_img.get_fdata()
    hippo_values = map_data[atlas_data == 1]
    amygdala_values = map_data[atlas_data == 2]

    hippo = df.loc[df["region_id"] == 1].iloc[0]
    amygdala = df.loc[df["region_id"] == 2].iloc[0]

    assert hippo["region_name"] == "hippocampus"
    assert hippo["voxel_count"] == 2
    assert hippo["mean"] == pytest.approx(np.nanmean(hippo_values))
    assert hippo["median"] == pytest.approx(np.nanmedian(hippo_values))
    assert hippo["std"] == pytest.approx(np.nanstd(hippo_values))

    assert amygdala["region_name"] == "amygdala"
    assert amygdala["voxel_count"] == 4
    assert amygdala["mean"] == pytest.approx(np.nanmean(amygdala_values))
    assert amygdala["median"] == pytest.approx(np.nanmedian(amygdala_values))
    assert amygdala["std"] == pytest.approx(np.nanstd(amygdala_values))


def test_resampling_aligns_to_atlas_grid() -> None:
    atlas_img = _atlas()
    parcellator = Parcellator(atlas_img, resampling="map_to_atlas")

    fine_affine = np.diag([0.5, 0.5, 0.5, 1.0])
    fine_map = nib.Nifti1Image(np.ones((4, 4, 4)) * 5.0, fine_affine)

    df = parcellator.parcellate(fine_map)
    assert df["mean"].tolist() == pytest.approx([5.0, 5.0])


def test_resampling_can_be_disabled() -> None:
    atlas_img = _atlas()
    parcellator = Parcellator(atlas_img, resampling="raise")

    coarse_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    coarse_map = nib.Nifti1Image(np.ones((1, 1, 1)), coarse_affine)

    with pytest.raises(ValueError):
        parcellator.parcellate(coarse_map)


def test_custom_statistics_override_defaults() -> None:
    atlas_img = _atlas()
    map_img = nib.Nifti1Image(
        np.arange(8, dtype=np.float32).reshape((2, 2, 2)), atlas_img.affine
    )

    parcellator = Parcellator(atlas_img, stat_functions={"max": np.nanmax})
    df = parcellator.parcellate(map_img, stat_functions={"min": np.nanmin})

    assert "mean" not in df.columns
    assert "median" not in df.columns
    assert "std" not in df.columns
    assert df["min"].tolist() == pytest.approx([1.0, 3.0])

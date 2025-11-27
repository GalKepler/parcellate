from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from parcellate import VolumetricParcellator
from parcellate.parcellation.volume import ParcellatorNotFittedError


def _atlas() -> nib.Nifti1Image:
    data = np.array(
        [
            [[0, 1], [1, 2]],
            [[0, 2], [2, 2]],
        ],
        dtype=np.int16,
    )
    return nib.Nifti1Image(data, np.eye(4))


def test_fit_and_transform_compute_basic_statistics() -> None:
    atlas_img = _atlas()
    scalar_data = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    scalar_img = nib.Nifti1Image(scalar_data, atlas_img.affine)

    parcellator = VolumetricParcellator(atlas_img)
    parcellator.fit(scalar_img)
    df = parcellator.transform(scalar_img)

    region1_values = scalar_data[np.asarray(atlas_img.get_fdata()) == 1]
    region2_values = scalar_data[np.asarray(atlas_img.get_fdata()) == 2]

    first = df.loc[df["index"] == 1].iloc[0]
    second = df.loc[df["index"] == 2].iloc[0]

    assert first["voxel_count"] == 2
    assert second["voxel_count"] == 4

    assert first["mean"] == pytest.approx(np.nanmean(region1_values))
    assert first["median"] == pytest.approx(np.nanmedian(region1_values))
    assert first["std"] == pytest.approx(np.nanstd(region1_values))
    assert first["volume_mm3"] == pytest.approx(2.0)

    assert second["mean"] == pytest.approx(np.nanmean(region2_values))
    assert second["median"] == pytest.approx(np.nanmedian(region2_values))
    assert second["std"] == pytest.approx(np.nanstd(region2_values))
    assert second["volume_mm3"] == pytest.approx(4.0)


def test_masked_atlas_excludes_voxels() -> None:
    atlas_img = _atlas()
    scalar_img = nib.Nifti1Image(
        np.ones((2, 2, 2), dtype=np.float32),
        atlas_img.affine,
    )
    mask = nib.Nifti1Image(
        np.array(
            [
                [[1, 1], [1, 1]],
                [[1, 1], [1, 0]],
            ],
            dtype=np.uint8,
        ),
        atlas_img.affine,
    )

    parcellator = VolumetricParcellator(atlas_img, mask=mask)
    parcellator.fit(scalar_img)
    df = parcellator.transform(scalar_img)

    second = df.loc[df["index"] == 2].iloc[0]
    assert second["voxel_count"] == 3
    assert second["volume_mm3"] == pytest.approx(3.0)


def test_custom_statistics_override_defaults() -> None:
    atlas_img = _atlas()
    scalar_img = nib.Nifti1Image(np.arange(8, dtype=np.float32).reshape((2, 2, 2)), atlas_img.affine)

    parcellator = VolumetricParcellator(atlas_img, stat_functions={"min": np.nanmin})
    parcellator.fit(scalar_img)
    df = parcellator.transform(scalar_img)

    assert set(df.columns) == {"index", "label", "min"}
    assert df["min"].tolist() == pytest.approx([1.0, 3.0])


def test_transform_requires_fit() -> None:
    atlas_img = _atlas()
    scalar_img = nib.Nifti1Image(np.ones((2, 2, 2)), atlas_img.affine)
    parcellator = VolumetricParcellator(atlas_img)

    with pytest.raises(ParcellatorNotFittedError):
        parcellator.transform(scalar_img)


def test_scalar_image_zero_std() -> None:
    atlas_img = _atlas()
    scalar_data = np.ones((2, 2, 2), dtype=np.float32)
    scalar_img = nib.Nifti1Image(scalar_data, atlas_img.affine)

    parcellator = VolumetricParcellator(atlas_img)
    parcellator.fit(scalar_img)
    df = parcellator.transform(scalar_img)

    first = df.loc[df["index"] == 1].iloc[0]
    second = df.loc[df["index"] == 2].iloc[0]

    assert first["std"] == 0.0
    assert second["std"] == 0.0


def test_empty_parcel_handling() -> None:
    atlas_img = _atlas()
    scalar_data = np.array(
        [
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[np.nan, np.nan], [np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    scalar_img = nib.Nifti1Image(scalar_data, atlas_img.affine)

    parcellator = VolumetricParcellator(atlas_img)
    parcellator.fit(scalar_img)
    df = parcellator.transform(scalar_img)

    first = df.loc[df["index"] == 1].iloc[0]
    second = df.loc[df["index"] == 2].iloc[0]

    assert first["voxel_count"] == 2
    assert second["voxel_count"] == 4

    assert np.isnan(first["mean"])
    assert np.isnan(first["median"])
    assert np.isnan(first["std"])
    assert first["volume_mm3"] == pytest.approx(2.0)

    assert np.isnan(second["mean"])
    assert np.isnan(second["median"])
    assert np.isnan(second["std"])
    assert second["volume_mm3"] == pytest.approx(4.0)


def test_no_valid_voxels_in_parcel() -> None:
    atlas_img = _atlas()
    scalar_data = np.array(
        [
            [[np.nan, 2.0], [3.0, 4.0]],
            [[5.0, np.nan], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    scalar_img = nib.Nifti1Image(scalar_data, atlas_img.affine)

    parcellator = VolumetricParcellator(atlas_img)
    parcellator.fit(scalar_img)
    df = parcellator.transform(scalar_img)

    region1_values = scalar_data[np.asarray(atlas_img.get_fdata()) == 1]
    region2_values = scalar_data[np.asarray(atlas_img.get_fdata()) == 2]

    first = df.loc[df["index"] == 1].iloc[0]
    second = df.loc[df["index"] == 2].iloc[0]

    assert first["voxel_count"] == 2
    assert second["voxel_count"] == 4

    assert first["mean"] == pytest.approx(np.nanmean(region1_values))
    assert first["median"] == pytest.approx(np.nanmedian(region1_values))
    assert first["std"] == pytest.approx(np.nanstd(region1_values))
    assert first["volume_mm3"] == pytest.approx(2.0)

    assert second["mean"] == pytest.approx(np.nanmean(region2_values))
    assert second["median"] == pytest.approx(np.nanmedian(region2_values))
    assert second["std"] == pytest.approx(np.nanstd(region2_values))
    assert second["volume_mm3"] == pytest.approx(4.0)


def test_atlas_is_filename() -> None:
    atlas_img = _atlas()
    atlas_path = "temp_atlas.nii"
    nib.save(atlas_img, atlas_path)

    scalar_data = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    scalar_img = nib.Nifti1Image(scalar_data, atlas_img.affine)

    parcellator = VolumetricParcellator(atlas_path)
    parcellator.fit(scalar_img)
    df = parcellator.transform(scalar_img)

    region1_values = scalar_data[np.asarray(atlas_img.get_fdata()) == 1]

    first = df.loc[df["index"] == 1].iloc[0]
    second = df.loc[df["index"] == 2].iloc[0]
    assert first["voxel_count"] == 2
    assert second["voxel_count"] == 4

    assert first["mean"] == pytest.approx(np.nanmean(region1_values))
    assert first["median"] == pytest.approx(np.nanmedian(region1_values))
    assert first["std"] == pytest.approx(np.nanstd(region1_values))
    assert first["volume_mm3"] == pytest.approx(2.0)

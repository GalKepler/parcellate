from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from parcellate import VolumetricParcellator
from parcellate.metrics.volume import BUILTIN_STATISTICS
from parcellate.parcellation.volume import AtlasShapeError, MissingLUTColumnsError, ParcellatorNotFittedError


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
    # volume_mm3 = sum of tissue intensities * voxel_volume
    assert first["volume_mm3"] == pytest.approx(np.nansum(region1_values))

    assert second["mean"] == pytest.approx(np.nanmean(region2_values))
    assert second["median"] == pytest.approx(np.nanmedian(region2_values))
    assert second["std"] == pytest.approx(np.nanstd(region2_values))
    assert second["volume_mm3"] == pytest.approx(np.nansum(region2_values))


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
    # With all NaN values, nansum returns 0.0, so volume is 0.0
    assert first["volume_mm3"] == pytest.approx(0.0)

    assert np.isnan(second["mean"])
    assert np.isnan(second["median"])
    assert np.isnan(second["std"])
    assert second["volume_mm3"] == pytest.approx(0.0)


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
    # volume_mm3 = nansum of tissue intensities * voxel_volume
    assert first["volume_mm3"] == pytest.approx(np.nansum(region1_values))

    assert second["mean"] == pytest.approx(np.nanmean(region2_values))
    assert second["median"] == pytest.approx(np.nanmedian(region2_values))
    assert second["std"] == pytest.approx(np.nanstd(region2_values))
    assert second["volume_mm3"] == pytest.approx(np.nansum(region2_values))


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
    # volume_mm3 = sum of tissue intensities * voxel_volume
    assert first["volume_mm3"] == pytest.approx(np.nansum(region1_values))


def test_lut_missing_columns() -> None:
    atlas_img = _atlas()
    lut = pd.DataFrame(
        {
            "some_other_column": ["Region 1", "Region 2", "Region 3"],
        },
        index=[0, 1, 2],
    )

    with pytest.raises(MissingLUTColumnsError):
        VolumetricParcellator(atlas_img, lut=lut)


def test_atlas_5d_raises_error() -> None:
    """5D+ atlas should still raise AtlasShapeError."""
    data = np.zeros((2, 2, 2, 2, 2), dtype=np.float32)
    atlas_img_5d = nib.Nifti1Image(data, np.eye(4))
    with pytest.raises(AtlasShapeError):
        VolumetricParcellator(atlas_img_5d)


def test_no_statistical_functions() -> None:
    atlas_img = _atlas()
    with pytest.raises(ValueError):
        vp = VolumetricParcellator(atlas_img, stat_functions={})
        vp._prepare_stat_functions()


def test_fallback_statistics() -> None:
    atlas_img = _atlas()
    vp = VolumetricParcellator(atlas_img)
    stats = vp._prepare_stat_functions(fallback=BUILTIN_STATISTICS)
    assert stats == BUILTIN_STATISTICS


def test_builtin_standard_masks() -> None:
    atlas_img = _atlas()
    scalar_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), atlas_img.affine)

    for mask_name in ["gm", "wm", "brain"]:
        parcellator = VolumetricParcellator(atlas_img, mask=mask_name)
        parcellator.fit(scalar_img)
        df = parcellator.transform(scalar_img)

        total_voxels = np.sum(np.asarray(atlas_img.get_fdata()) > 0)
        voxel_count = df["voxel_count"].sum()
        assert voxel_count <= total_voxels


def test_labels_as_list() -> None:
    atlas_img = _atlas()
    scalar_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), atlas_img.affine)

    labels = [1, 2]
    parcellator = VolumetricParcellator(atlas_img, labels=labels)
    parcellator.fit(scalar_img)
    df = parcellator.transform(scalar_img)

    first = df.loc[df["index"] == 1].iloc[0]
    second = df.loc[df["index"] == 2].iloc[0]

    assert first["label"] == "1"
    assert second["label"] == "2"


def test_labels_as_dict() -> None:
    atlas_img = _atlas()
    scalar_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), atlas_img.affine)

    labels = {1: "Region_One", 2: "Region_Two"}
    parcellator = VolumetricParcellator(atlas_img, labels=labels)
    parcellator.fit(scalar_img)
    df = parcellator.transform(scalar_img)

    first = df.loc[df["index"] == 1].iloc[0]
    second = df.loc[df["index"] == 2].iloc[0]

    assert first["label"] == "1"
    assert second["label"] == "2"


def test_labels_from_lut() -> None:
    atlas_img = _atlas()
    scalar_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), atlas_img.affine)

    lut = pd.DataFrame({
        "index": [0, 1, 2],
        "label": ["Background", "Region_A", "Region_B"],
    })

    parcellator = VolumetricParcellator(atlas_img, lut=lut)
    parcellator.fit(scalar_img)
    df = parcellator.transform(scalar_img)

    first = df.loc[df["index"] == 1].iloc[0]
    second = df.loc[df["index"] == 2].iloc[0]

    assert first["label"] == "Region_A"
    assert second["label"] == "Region_B"
    assert isinstance(parcellator.lut, pd.DataFrame)
    assert parcellator.regions == (1, 2)


def test_resample_to_atlas() -> None:
    atlas_img = _atlas()
    scalar_data = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float32,
    )
    scalar_img = nib.Nifti1Image(scalar_data, np.eye(4) * 2)  # Different voxel size

    parcellator = VolumetricParcellator(atlas_img, resampling_target="atlas")
    parcellator.fit(scalar_img)

    assert parcellator.ref_img == atlas_img
    assert parcellator._prepared_scalar_img.shape == atlas_img.shape


def test_region_not_in_index() -> None:
    atlas_img = _atlas()
    scalar_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), atlas_img.affine)
    lut = pd.DataFrame({
        "index": [0, 1, 2],
        "label": ["Background", "Region_A", "Region_B"],
    })
    parcellator = VolumetricParcellator(atlas_img, labels=[3], lut=lut)  # Region 3 does not exist in atlas
    parcellator.fit(scalar_img)
    df = parcellator.transform(scalar_img)

    assert 3 not in df["index"].values
    assert df.columns.tolist() == ["index", "label"]


def test_stat_functions_invalid_type() -> None:
    """Test that invalid stat_functions type raises MissingStatisticalFunctionError."""
    from parcellate.parcellation.volume import MissingStatisticalFunctionError

    atlas_img = _atlas()

    # Passing an invalid type (e.g., an integer) should raise an error
    # Note: strings are Sequences so they go through a different code path
    with pytest.raises(MissingStatisticalFunctionError) as excinfo:
        VolumetricParcellator(atlas_img, stat_functions=42)

    assert "must be a Mapping or Sequence" in str(excinfo.value)


def test_stat_functions_invalid_sequence_contents() -> None:
    """Test that sequence with non-Statistic items raises MissingStatisticalFunctionError."""
    from parcellate.parcellation.volume import MissingStatisticalFunctionError

    atlas_img = _atlas()

    # Passing a sequence with non-Statistic items should raise an error
    with pytest.raises(MissingStatisticalFunctionError) as excinfo:
        VolumetricParcellator(atlas_img, stat_functions=["not", "statistics"])

    assert "must contain Statistic instances" in str(excinfo.value)


def test_stat_functions_as_sequence_of_statistics() -> None:
    """Test that valid sequence of Statistic objects works."""
    from parcellate.metrics.base import Statistic

    atlas_img = _atlas()
    scalar_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), atlas_img.affine)

    # Create a list of Statistic objects
    stats = [
        Statistic(name="custom_mean", function=np.nanmean),
        Statistic(name="custom_std", function=np.nanstd),
    ]

    parcellator = VolumetricParcellator(atlas_img, stat_functions=stats)
    parcellator.fit(scalar_img)
    df = parcellator.transform(scalar_img)

    assert "custom_mean" in df.columns
    assert "custom_std" in df.columns


def test_resampling_target_labels() -> None:
    """Test that resampling_target='labels' works correctly."""
    atlas_img = _atlas()
    scalar_data = np.ones((4, 4, 4), dtype=np.float32)  # Different size than atlas
    scalar_img = nib.Nifti1Image(scalar_data, np.eye(4) * 0.5)  # Different voxel size

    parcellator = VolumetricParcellator(atlas_img, resampling_target="labels")
    parcellator.fit(scalar_img)

    # Reference should be the atlas
    assert parcellator.ref_img == atlas_img


def test_mask_as_none_explicitly() -> None:
    """Test that explicitly passing mask=None works."""
    atlas_img = _atlas()
    scalar_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), atlas_img.affine)

    parcellator = VolumetricParcellator(atlas_img, mask=None)
    assert parcellator.mask is None

    parcellator.fit(scalar_img)
    df = parcellator.transform(scalar_img)

    # Should work without mask
    assert len(df) == 2  # Two regions


def test_lut_from_tsv_file(tmp_path) -> None:
    """Test loading LUT from a TSV file."""
    atlas_img = _atlas()
    scalar_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), atlas_img.affine)

    # Create a TSV file
    lut_path = tmp_path / "lut.tsv"
    lut_df = pd.DataFrame({
        "index": [0, 1, 2],
        "label": ["Background", "Region_A", "Region_B"],
    })
    lut_df.to_csv(lut_path, sep="\t", index=False)

    parcellator = VolumetricParcellator(atlas_img, lut=lut_path)
    parcellator.fit(scalar_img)
    df = parcellator.transform(scalar_img)

    assert df.loc[df["index"] == 1, "label"].iloc[0] == "Region_A"
    assert df.loc[df["index"] == 2, "label"].iloc[0] == "Region_B"


def test_background_label_custom() -> None:
    """Test custom background label."""
    # Create atlas where 1 is background instead of 0
    data = np.array(
        [
            [[1, 2], [2, 3]],
            [[1, 3], [3, 3]],
        ],
        dtype=np.int16,
    )
    atlas_img = nib.Nifti1Image(data, np.eye(4))
    scalar_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), atlas_img.affine)

    parcellator = VolumetricParcellator(atlas_img, background_label=1)
    parcellator.fit(scalar_img)
    df = parcellator.transform(scalar_img)

    # Should only have regions 2 and 3, not 1
    assert 1 not in df["index"].values
    assert 2 in df["index"].values
    assert 3 in df["index"].values


def test_transform_with_same_path_as_fit_reuses_prepared_image(mocker) -> None:
    """Test that transform() with the same path as fit() reuses the prepared image."""
    atlas_img = _atlas()
    scalar_img_path = "scalar.nii"
    scalar_img = nib.Nifti1Image(np.ones((2, 2, 2)), atlas_img.affine)
    nib.save(scalar_img, scalar_img_path)

    parcellator = VolumetricParcellator(atlas_img)
    parcellator.fit(scalar_img_path)

    prepare_map_spy = mocker.spy(parcellator, "_prepare_map")

    df = parcellator.transform(scalar_img_path)

    assert df is not None
    # _prepare_map should not be called in transform because the image is cached
    assert prepare_map_spy.call_count == 0


def test_transform_with_different_path_resamples(mocker) -> None:
    """Test that transform() with a different path resamples the image."""
    atlas_img = _atlas()
    scalar_img_path_1 = "scalar1.nii"
    scalar_img_path_2 = "scalar2.nii"
    scalar_img_1 = nib.Nifti1Image(np.ones((2, 2, 2)), atlas_img.affine)
    scalar_img_2 = nib.Nifti1Image(np.zeros((2, 2, 2)), atlas_img.affine)
    nib.save(scalar_img_1, scalar_img_path_1)
    nib.save(scalar_img_2, scalar_img_path_2)

    parcellator = VolumetricParcellator(atlas_img)
    parcellator.fit(scalar_img_path_1)

    prepare_map_spy = mocker.spy(parcellator, "_prepare_map")

    df = parcellator.transform(scalar_img_path_2)

    assert df is not None
    # _prepare_map should be called because the image is different
    assert prepare_map_spy.call_count == 1


# ---------------------------------------------------------------------------
# 4D probabilistic atlas tests
# ---------------------------------------------------------------------------


def _prob_atlas_2x2x2x2() -> nib.Nifti1Image:
    """Create a small 2x2x2x2 probabilistic atlas with two regions."""
    # Volume 0 (region 1): high probability in voxels [0,0,0] and [0,1,0]
    # Volume 1 (region 2): high probability in voxels [1,0,1] and [1,1,1]
    data = np.zeros((2, 2, 2, 2), dtype=np.float32)
    data[0, 0, 0, 0] = 0.9
    data[0, 1, 0, 0] = 0.8
    data[1, 0, 1, 1] = 0.7
    data[1, 1, 1, 1] = 0.6
    return nib.Nifti1Image(data, np.eye(4))


def test_4d_atlas_accepted() -> None:
    """A 4D atlas should be accepted and detected as probabilistic."""
    atlas_img = _prob_atlas_2x2x2x2()
    vp = VolumetricParcellator(atlas_img)

    assert vp._is_probabilistic is True
    assert vp.regions == (1, 2)


def test_probabilistic_fit_transform() -> None:
    """End-to-end fit+transform with a 4D probabilistic atlas."""
    atlas_img = _prob_atlas_2x2x2x2()
    scalar_data = np.ones((2, 2, 2), dtype=np.float32) * 3.0
    scalar_img = nib.Nifti1Image(scalar_data, atlas_img.affine)

    vp = VolumetricParcellator(atlas_img)
    vp.fit(scalar_img)
    df = vp.transform(scalar_img)

    assert set(df["index"].tolist()) == {1, 2}
    r1 = df.loc[df["index"] == 1].iloc[0]
    r2 = df.loc[df["index"] == 2].iloc[0]
    # Default threshold=0.0: both non-zero voxels in each volume should be included
    assert r1["voxel_count"] == 2
    assert r2["voxel_count"] == 2
    assert r1["mean"] == pytest.approx(3.0)
    assert r2["mean"] == pytest.approx(3.0)


def test_probabilistic_overlap() -> None:
    """A voxel with high probability in two regions appears in both stats."""
    # Overlap voxel [0,0,0] has high prob in both volumes
    data = np.zeros((2, 2, 2, 2), dtype=np.float32)
    data[0, 0, 0, 0] = 0.9  # region 1
    data[0, 0, 0, 1] = 0.8  # region 2 — same spatial voxel
    data[1, 1, 1, 0] = 0.7  # region 1 only
    atlas_img = nib.Nifti1Image(data, np.eye(4))

    scalar_data = np.array(
        [[[10.0, 5.0], [5.0, 5.0]], [[5.0, 5.0], [5.0, 20.0]]],
        dtype=np.float32,
    )
    scalar_img = nib.Nifti1Image(scalar_data, atlas_img.affine)

    vp = VolumetricParcellator(atlas_img)
    vp.fit(scalar_img)
    df = vp.transform(scalar_img)

    r1 = df.loc[df["index"] == 1].iloc[0]
    r2 = df.loc[df["index"] == 2].iloc[0]
    # Overlap voxel [0,0,0]=10.0 must appear in both regions
    assert r1["voxel_count"] == 2  # [0,0,0] and [1,1,1]
    assert r2["voxel_count"] == 1  # [0,0,0] only
    assert r1["mean"] == pytest.approx(np.mean([10.0, 20.0]))
    assert r2["mean"] == pytest.approx(10.0)


def test_probabilistic_with_lut() -> None:
    """LUT label names appear in output; 1-based index mapping works."""
    atlas_img = _prob_atlas_2x2x2x2()
    lut = pd.DataFrame({
        "index": [1, 2],
        "label": ["TractA", "TractB"],
    })
    scalar_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), atlas_img.affine)

    vp = VolumetricParcellator(atlas_img, lut=lut)
    vp.fit(scalar_img)
    df = vp.transform(scalar_img)

    assert df.loc[df["index"] == 1, "label"].iloc[0] == "TractA"
    assert df.loc[df["index"] == 2, "label"].iloc[0] == "TractB"


def test_probabilistic_with_mask() -> None:
    """External mask correctly zeros out probability across all volumes."""
    atlas_img = _prob_atlas_2x2x2x2()
    # Mask out voxel [0,0,0] — it has high prob in region 1
    mask_data = np.ones((2, 2, 2), dtype=np.uint8)
    mask_data[0, 0, 0] = 0
    mask_img = nib.Nifti1Image(mask_data, atlas_img.affine)
    scalar_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), atlas_img.affine)

    vp = VolumetricParcellator(atlas_img, mask=mask_img)
    vp.fit(scalar_img)
    df = vp.transform(scalar_img)

    r1 = df.loc[df["index"] == 1].iloc[0]
    # Voxel [0,0,0] masked out → only [0,1,0] remains for region 1
    assert r1["voxel_count"] == 1


def test_probabilistic_threshold_excludes_low() -> None:
    """Voxels at or below threshold are excluded from region masks."""
    data = np.zeros((2, 2, 2, 1), dtype=np.float32)
    data[0, 0, 0, 0] = 0.5  # exactly at threshold — excluded
    data[0, 1, 0, 0] = 0.51  # just above — included
    data[1, 0, 0, 0] = 0.2  # well below — excluded
    atlas_img = nib.Nifti1Image(data, np.eye(4))
    scalar_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), atlas_img.affine)

    vp = VolumetricParcellator(atlas_img, atlas_threshold=0.5)
    vp.fit(scalar_img)
    df = vp.transform(scalar_img)

    r1 = df.loc[df["index"] == 1].iloc[0]
    assert r1["voxel_count"] == 1  # only the 0.51 voxel


def test_probabilistic_default_threshold_zero() -> None:
    """Default threshold=0.0 includes any non-zero probability voxel."""
    data = np.zeros((2, 2, 2, 1), dtype=np.float32)
    data[0, 0, 0, 0] = 0.01  # tiny but non-zero
    data[0, 1, 0, 0] = 0.99
    atlas_img = nib.Nifti1Image(data, np.eye(4))
    scalar_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), atlas_img.affine)

    vp = VolumetricParcellator(atlas_img)  # atlas_threshold=0.0 by default
    vp.fit(scalar_img)
    df = vp.transform(scalar_img)

    r1 = df.loc[df["index"] == 1].iloc[0]
    assert r1["voxel_count"] == 2  # both non-zero voxels included


def test_3d_atlas_unaffected() -> None:
    """Existing 3D discrete atlas behaviour is unchanged by the 4D feature."""
    atlas_img = _atlas()
    scalar_data = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    scalar_img = nib.Nifti1Image(scalar_data, atlas_img.affine)

    vp = VolumetricParcellator(atlas_img)
    assert vp._is_probabilistic is False
    vp.fit(scalar_img)
    df = vp.transform(scalar_img)

    assert set(df["index"].tolist()) == {1, 2}
    r1 = df.loc[df["index"] == 1].iloc[0]
    r2 = df.loc[df["index"] == 2].iloc[0]
    assert r1["voxel_count"] == 2
    assert r2["voxel_count"] == 4

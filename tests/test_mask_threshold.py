"""Tests for mask threshold support in VolumetricParcellator."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from parcellate import VolumetricParcellator


def _make_atlas() -> nib.Nifti1Image:
    """3-D atlas with two regions (1, 2) and background (0).

    Indexing is data[x, y, z] (C order):
        x=0: [[0, 1], [2, 2]]   → (x=0,y=0,z=0)=0, (x=0,y=0,z=1)=1,
                                   (x=0,y=1,z=0)=2, (x=0,y=1,z=1)=2
        x=1: [[0, 2], [2, 2]]   → (x=1,y=0,z=0)=0, (x=1,y=0,z=1)=2,
                                   (x=1,y=1,z=0)=2, (x=1,y=1,z=1)=2

    region 1: 1 voxel  at (0,0,1)
    region 2: 5 voxels at (0,1,0), (0,1,1), (1,0,1), (1,1,0), (1,1,1)
    background: 2 voxels at (0,0,0), (1,0,0)
    """
    data = np.array(
        [
            [[0, 1], [2, 2]],
            [[0, 2], [2, 2]],
        ],
        dtype=np.int16,
    )
    return nib.Nifti1Image(data, np.eye(4))


def _make_scalar(value: float = 1.0) -> nib.Nifti1Image:
    """Uniform scalar image filled with *value*."""
    atlas = _make_atlas()
    data = np.full((2, 2, 2), value, dtype=np.float32)
    return nib.Nifti1Image(data, atlas.affine)


def _make_prob_mask(values: list[float]) -> nib.Nifti1Image:
    """Create a probability mask in the same space as the atlas."""
    assert len(values) == 8, "Need exactly 8 values for a 2x2x2 mask"
    data = np.array(values, dtype=np.float32).reshape((2, 2, 2))
    atlas = _make_atlas()
    return nib.Nifti1Image(data, atlas.affine)


class TestMaskThresholdParcellator:
    """Test threshold is correctly applied in _apply_mask_to_atlas."""

    def test_default_threshold_zero_passes_nonzero_voxels(self) -> None:
        atlas = _make_atlas()
        scalar = _make_scalar(1.0)
        mask = _make_prob_mask([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        vp = VolumetricParcellator(atlas, mask=mask, mask_threshold=0.0)
        vp.fit(scalar)
        df = vp.transform(scalar)
        assert df.loc[df["index"] == 1, "voxel_count"].iloc[0] == 1
        assert df.loc[df["index"] == 2, "voxel_count"].iloc[0] == 5

    def test_threshold_excludes_low_probability_voxels(self) -> None:
        atlas = _make_atlas()
        scalar = _make_scalar(1.0)
        probs = [0.0, 0.8, 0.9, 0.9, 0.0, 0.9, 0.9, 0.3]
        mask = _make_prob_mask(probs)
        vp = VolumetricParcellator(atlas, mask=mask, mask_threshold=0.5)
        vp.fit(scalar)
        df = vp.transform(scalar)
        assert df.loc[df["index"] == 1, "voxel_count"].iloc[0] == 1
        assert df.loc[df["index"] == 2, "voxel_count"].iloc[0] == 4

    def test_threshold_at_exactly_boundary_is_exclusive(self) -> None:
        atlas = _make_atlas()
        scalar = _make_scalar(1.0)
        probs = [0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5]
        mask = _make_prob_mask(probs)
        vp = VolumetricParcellator(atlas, mask=mask, mask_threshold=0.5)
        vp.fit(scalar)
        df = vp.transform(scalar)
        assert df.loc[df["index"] == 1, "voxel_count"].iloc[0] == 0
        assert df.loc[df["index"] == 2, "voxel_count"].iloc[0] == 0

    def test_threshold_one_excludes_all_voxels(self) -> None:
        atlas = _make_atlas()
        scalar = _make_scalar(1.0)
        mask = _make_prob_mask([0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        vp = VolumetricParcellator(atlas, mask=mask, mask_threshold=1.0)
        vp.fit(scalar)
        df = vp.transform(scalar)
        assert df.loc[df["index"] == 1, "voxel_count"].iloc[0] == 0
        assert df.loc[df["index"] == 2, "voxel_count"].iloc[0] == 0

    def test_backward_compatibility_bool_cast_equivalent(self) -> None:
        atlas = _make_atlas()
        scalar = _make_scalar(2.5)
        probs = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        mask = _make_prob_mask(probs)

        vp_new = VolumetricParcellator(atlas, mask=mask, mask_threshold=0.0)
        vp_new.fit(scalar)
        df_new = vp_new.transform(scalar)

        atlas_data = np.asarray(atlas.get_fdata()).copy()
        mask_data = np.asarray(mask.get_fdata()).astype(bool)
        atlas_data[~mask_data] = 0
        masked_atlas = nib.Nifti1Image(atlas_data.astype(np.int16), atlas.affine)
        vp_ref = VolumetricParcellator(masked_atlas)
        vp_ref.fit(scalar)
        df_ref = vp_ref.transform(scalar)

        for region_id in (1, 2):
            new_count = df_new.loc[df_new["index"] == region_id, "voxel_count"].iloc[0]
            ref_count = df_ref.loc[df_ref["index"] == region_id, "voxel_count"].iloc[0]
            assert new_count == ref_count, f"Region {region_id}: {new_count} != {ref_count}"

    def test_mask_threshold_stored_as_attribute(self) -> None:
        atlas = _make_atlas()
        vp = VolumetricParcellator(atlas, mask_threshold=0.3)
        assert vp.mask_threshold == pytest.approx(0.3)

    def test_default_mask_threshold_is_zero(self) -> None:
        atlas = _make_atlas()
        vp = VolumetricParcellator(atlas)
        assert vp.mask_threshold == 0.0

    def test_mask_threshold_cast_to_float(self) -> None:
        atlas = _make_atlas()
        vp = VolumetricParcellator(atlas, mask_threshold=1)
        assert isinstance(vp.mask_threshold, float)
        assert vp.mask_threshold == 1.0

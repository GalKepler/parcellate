"""Tests for mask threshold support across the parcellate stack.

Covers:
- VolumetricParcellator: threshold behaviour with probability masks
- _mask_threshold_label: compact label formatting
- Config parsing: mask_threshold read from TOML data
- Output filenames: maskthr-* entity when threshold is non-default
- Sidecar JSON: mask_threshold recorded in provenance
- Backward compatibility: default threshold=0.0 identical to original bool-cast
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from parcellate import VolumetricParcellator
from parcellate.interfaces.utils import _mask_threshold_label, write_parcellation_sidecar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Create a probability mask in the same space as the atlas.

    *values* must have exactly 8 elements corresponding to the ravelled voxels
    of a (2, 2, 2) volume in C order.
    """
    assert len(values) == 8, "Need exactly 8 values for a 2x2x2 mask"
    data = np.array(values, dtype=np.float32).reshape((2, 2, 2))
    atlas = _make_atlas()
    return nib.Nifti1Image(data, atlas.affine)


# ---------------------------------------------------------------------------
# VolumetricParcellator — threshold behaviour
# ---------------------------------------------------------------------------


class TestMaskThresholdParcellator:
    """Test threshold is correctly applied in _apply_mask_to_atlas."""

    def test_default_threshold_zero_passes_nonzero_voxels(self) -> None:
        """Default threshold=0.0: any voxel with mask>0 is included."""
        atlas = _make_atlas()
        scalar = _make_scalar(1.0)
        # All voxels have mask value 1 → all included
        mask = _make_prob_mask([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        vp = VolumetricParcellator(atlas, mask=mask, mask_threshold=0.0)
        vp.fit(scalar)
        df = vp.transform(scalar)
        # Region 1: 1 voxel at (0,0,1); region 2: 5 voxels (see _make_atlas docstring)
        assert df.loc[df["index"] == 1, "voxel_count"].iloc[0] == 1
        assert df.loc[df["index"] == 2, "voxel_count"].iloc[0] == 5

    def test_threshold_excludes_low_probability_voxels(self) -> None:
        """Voxels with mask value <= threshold are zeroed out of atlas."""
        atlas = _make_atlas()
        scalar = _make_scalar(1.0)
        # Ravelled (C order) probabilities for 2x2x2 volume:
        # index → atlas position → atlas label:
        #   [0,0,0]=0.0 → bg=0
        #   [0,0,1]=0.8 → reg1=1  mask 0.8 > 0.5 ✓ included
        #   [0,1,0]=0.9 → reg2=2  mask 0.9 > 0.5 ✓
        #   [0,1,1]=0.9 → reg2=2  mask 0.9 > 0.5 ✓
        #   [1,0,0]=0.0 → bg=0
        #   [1,0,1]=0.9 → reg2=2  mask 0.9 > 0.5 ✓
        #   [1,1,0]=0.9 → reg2=2  mask 0.9 > 0.5 ✓
        #   [1,1,1]=0.3 → reg2=2  mask 0.3 ≤ 0.5 ✗ excluded
        probs = [0.0, 0.8, 0.9, 0.9, 0.0, 0.9, 0.9, 0.3]
        mask = _make_prob_mask(probs)

        vp = VolumetricParcellator(atlas, mask=mask, mask_threshold=0.5)
        vp.fit(scalar)
        df = vp.transform(scalar)

        # Region 1: 1 voxel at (0,0,1) with mask 0.8 > 0.5 → included
        assert df.loc[df["index"] == 1, "voxel_count"].iloc[0] == 1
        # Region 2: 5 voxels originally, but (1,1,1) mask=0.3 ≤ 0.5 → 4 included
        assert df.loc[df["index"] == 2, "voxel_count"].iloc[0] == 4

    def test_threshold_at_exactly_boundary_is_exclusive(self) -> None:
        """Voxels equal to the threshold are excluded (strict > comparison)."""
        atlas = _make_atlas()
        scalar = _make_scalar(1.0)
        # All region voxels have exactly mask=0.5; threshold=0.5 → all excluded
        probs = [0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5]
        mask = _make_prob_mask(probs)

        vp = VolumetricParcellator(atlas, mask=mask, mask_threshold=0.5)
        vp.fit(scalar)
        df = vp.transform(scalar)

        # All voxels excluded: voxel_count should be 0 for both regions
        assert df.loc[df["index"] == 1, "voxel_count"].iloc[0] == 0
        assert df.loc[df["index"] == 2, "voxel_count"].iloc[0] == 0

    def test_threshold_one_excludes_all_voxels(self) -> None:
        """Threshold=1.0 excludes every voxel (none can be strictly > 1)."""
        atlas = _make_atlas()
        scalar = _make_scalar(1.0)
        mask = _make_prob_mask([0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])

        vp = VolumetricParcellator(atlas, mask=mask, mask_threshold=1.0)
        vp.fit(scalar)
        df = vp.transform(scalar)

        assert df.loc[df["index"] == 1, "voxel_count"].iloc[0] == 0
        assert df.loc[df["index"] == 2, "voxel_count"].iloc[0] == 0

    def test_backward_compatibility_bool_cast_equivalent(self) -> None:
        """Default threshold=0.0 produces the same result as the old .astype(bool) path."""
        atlas = _make_atlas()
        scalar = _make_scalar(2.5)
        # Binary mask: some 0s, some 1s
        probs = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        mask = _make_prob_mask(probs)

        # New path (threshold=0.0)
        vp_new = VolumetricParcellator(atlas, mask=mask, mask_threshold=0.0)
        vp_new.fit(scalar)
        df_new = vp_new.transform(scalar)

        # Old path equivalent: apply mask manually with bool cast
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
        """mask_threshold is stored on the parcellator instance."""
        atlas = _make_atlas()
        vp = VolumetricParcellator(atlas, mask_threshold=0.3)
        assert vp.mask_threshold == pytest.approx(0.3)

    def test_default_mask_threshold_is_zero(self) -> None:
        """Default mask_threshold is 0.0."""
        atlas = _make_atlas()
        vp = VolumetricParcellator(atlas)
        assert vp.mask_threshold == 0.0

    def test_mask_threshold_cast_to_float(self) -> None:
        """mask_threshold is coerced to float even when an int is passed."""
        atlas = _make_atlas()
        vp = VolumetricParcellator(atlas, mask_threshold=1)
        assert isinstance(vp.mask_threshold, float)
        assert vp.mask_threshold == 1.0


# ---------------------------------------------------------------------------
# _mask_threshold_label
# ---------------------------------------------------------------------------


class TestMaskThresholdLabel:
    """Unit tests for the _mask_threshold_label helper."""

    def test_zero_returns_none(self) -> None:
        """Default threshold produces no filename entity."""
        assert _mask_threshold_label(0.0) is None

    def test_half_returns_50(self) -> None:
        """0.5 → '50' (integer percentage)."""
        assert _mask_threshold_label(0.5) == "50"

    def test_quarter_returns_25(self) -> None:
        """0.25 → '25'."""
        assert _mask_threshold_label(0.25) == "25"

    def test_full_returns_100(self) -> None:
        """1.0 → '100'."""
        assert _mask_threshold_label(1.0) == "100"

    def test_non_clean_fraction_uses_p_separator(self) -> None:
        """Non-integer percentage uses float repr with '.' replaced by 'p'."""
        # 0.123 * 100 = 12.299... (not a whole number) → "0p123"
        label = _mask_threshold_label(0.123)
        assert "p" in label
        assert "." not in label

    def test_0_3_returns_correct_label(self) -> None:
        """0.3 → '30' (30% is a whole number)."""
        assert _mask_threshold_label(0.3) == "30"

    def test_0_1_returns_correct_label(self) -> None:
        """0.1 → '10'."""
        assert _mask_threshold_label(0.1) == "10"


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestMaskThresholdConfigParsing:
    """Test mask_threshold is read from TOML data in load_config."""

    def test_cat12_load_config_reads_mask_threshold(self, tmp_path: Path) -> None:
        """mask_threshold key in TOML is correctly parsed into Cat12Config."""
        import argparse

        from parcellate.interfaces.cat12.cat12 import load_config

        toml_content = f"""
input_root = "{tmp_path}"
output_dir = "{tmp_path / "out"}"
mask_threshold = 0.5
""".strip()
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        args = argparse.Namespace(
            config=config_file,
            input_root=None,
            output_dir=None,
            atlas_config=None,
            subjects=None,
            sessions=None,
            mask=None,
            mask_threshold=None,
            force=False,
            log_level=None,
            n_jobs=None,
            n_procs=None,
        )
        config = load_config(args)
        assert config.mask_threshold == pytest.approx(0.5)

    def test_cat12_load_config_defaults_to_zero(self, tmp_path: Path) -> None:
        """When mask_threshold is absent from TOML, default is 0.0."""
        import argparse

        from parcellate.interfaces.cat12.cat12 import load_config

        toml_content = f"""
input_root = "{tmp_path}"
output_dir = "{tmp_path / "out"}"
""".strip()
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        args = argparse.Namespace(
            config=config_file,
            input_root=None,
            output_dir=None,
            atlas_config=None,
            subjects=None,
            sessions=None,
            mask=None,
            mask_threshold=None,
            force=False,
            log_level=None,
            n_jobs=None,
            n_procs=None,
        )
        config = load_config(args)
        assert config.mask_threshold == 0.0

    def test_qsirecon_load_config_reads_mask_threshold(self, tmp_path: Path) -> None:
        """mask_threshold key in TOML is correctly parsed into QSIReconConfig."""
        import argparse

        from parcellate.interfaces.qsirecon.qsirecon import load_config

        toml_content = f"""
input_root = "{tmp_path}"
output_dir = "{tmp_path / "out"}"
mask_threshold = 0.3
""".strip()
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        args = argparse.Namespace(
            config=config_file,
            input_root=None,
            output_dir=None,
            atlas_config=None,
            subjects=None,
            sessions=None,
            mask=None,
            mask_threshold=None,
            force=False,
            log_level=None,
            n_jobs=None,
            n_procs=None,
        )
        config = load_config(args)
        assert config.mask_threshold == pytest.approx(0.3)

    def test_cli_mask_threshold_arg_overrides_toml(self, tmp_path: Path) -> None:
        """CLI --mask-threshold overrides the TOML value."""
        import argparse

        from parcellate.interfaces.cat12.cat12 import load_config

        toml_content = f"""
input_root = "{tmp_path}"
output_dir = "{tmp_path / "out"}"
mask_threshold = 0.5
""".strip()
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        args = argparse.Namespace(
            config=config_file,
            input_root=None,
            output_dir=None,
            atlas_config=None,
            subjects=None,
            sessions=None,
            mask=None,
            mask_threshold=0.7,  # CLI overrides TOML 0.5
            force=False,
            log_level=None,
            n_jobs=None,
            n_procs=None,
        )
        config = load_config(args)
        assert config.mask_threshold == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------------


class TestMaskThresholdOutputFilenames:
    """Test maskthr-* entity in output filenames."""

    def test_cat12_default_threshold_no_entity(self, tmp_path: Path) -> None:
        """No maskthr entity when threshold is 0.0 (default)."""
        from parcellate.interfaces.cat12.cat12 import _build_output_path
        from parcellate.interfaces.cat12.models import ScalarMapDefinition, SubjectContext, TissueType
        from parcellate.interfaces.models import AtlasDefinition

        context = SubjectContext(subject_id="01")
        atlas = AtlasDefinition(name="TestAtlas", nifti_path=tmp_path / "atlas.nii.gz", space="MNI")
        scalar_map = ScalarMapDefinition(
            name="gm",
            nifti_path=tmp_path / "gm.nii.gz",
            space="MNI",
            tissue_type=TissueType.GM,
        )
        path = _build_output_path(context, atlas, scalar_map, tmp_path, mask="gm", mask_threshold=0.0)
        assert "maskthr" not in path.name

    def test_cat12_nondefault_threshold_adds_entity(self, tmp_path: Path) -> None:
        """maskthr entity added when threshold is non-default."""
        from parcellate.interfaces.cat12.cat12 import _build_output_path
        from parcellate.interfaces.cat12.models import ScalarMapDefinition, SubjectContext, TissueType
        from parcellate.interfaces.models import AtlasDefinition

        context = SubjectContext(subject_id="01")
        atlas = AtlasDefinition(name="TestAtlas", nifti_path=tmp_path / "atlas.nii.gz", space="MNI")
        scalar_map = ScalarMapDefinition(
            name="gm",
            nifti_path=tmp_path / "gm.nii.gz",
            space="MNI",
            tissue_type=TissueType.GM,
        )
        path = _build_output_path(context, atlas, scalar_map, tmp_path, mask="gm", mask_threshold=0.5)
        assert "maskthr-50" in path.name

    def test_qsirecon_nondefault_threshold_adds_entity(self, tmp_path: Path) -> None:
        """maskthr entity added in QSIRecon output paths."""
        from parcellate.interfaces.models import AtlasDefinition
        from parcellate.interfaces.qsirecon.models import ScalarMapDefinition, SubjectContext
        from parcellate.interfaces.qsirecon.qsirecon import _build_output_path

        context = SubjectContext(subject_id="01")
        atlas = AtlasDefinition(name="TestAtlas", nifti_path=tmp_path / "atlas.nii.gz", space="MNI")
        scalar_map = ScalarMapDefinition(
            name="FA",
            nifti_path=tmp_path / "FA.nii.gz",
            space="MNI",
            param="FA",
            recon_workflow="mrtrix",
        )
        path = _build_output_path(context, atlas, scalar_map, tmp_path, mask=None, mask_threshold=0.25)
        assert "maskthr-25" in path.name

    def test_cat12_25pct_threshold_label(self, tmp_path: Path) -> None:
        """0.25 threshold → maskthr-25 in filename."""
        from parcellate.interfaces.cat12.cat12 import _build_output_path
        from parcellate.interfaces.cat12.models import ScalarMapDefinition, SubjectContext, TissueType
        from parcellate.interfaces.models import AtlasDefinition

        context = SubjectContext(subject_id="02")
        atlas = AtlasDefinition(name="Atlas", nifti_path=tmp_path / "atlas.nii.gz", space="MNI")
        scalar_map = ScalarMapDefinition(
            name="wm",
            nifti_path=tmp_path / "wm.nii.gz",
            space="MNI",
            tissue_type=TissueType.WM,
        )
        path = _build_output_path(context, atlas, scalar_map, tmp_path, mask="wm", mask_threshold=0.25)
        assert "maskthr-25" in path.name


# ---------------------------------------------------------------------------
# Sidecar JSON
# ---------------------------------------------------------------------------


class TestMaskThresholdSidecar:
    """Test mask_threshold is recorded in the JSON sidecar."""

    def test_sidecar_contains_mask_threshold(self, tmp_path: Path) -> None:
        """write_parcellation_sidecar writes mask_threshold to JSON."""
        tsv_path = tmp_path / "parc.tsv"
        tsv_path.write_text("index\tlabel\n1\tregion1\n")

        write_parcellation_sidecar(
            tsv_path=tsv_path,
            original_file=tmp_path / "scalar.nii.gz",
            atlas_name="TestAtlas",
            atlas_image=tmp_path / "atlas.nii.gz",
            atlas_lut=None,
            mask="gm",
            mask_threshold=0.5,
        )

        json_path = tsv_path.with_suffix(".json")
        assert json_path.exists()
        sidecar = json.loads(json_path.read_text())
        assert "mask_threshold" in sidecar
        assert sidecar["mask_threshold"] == pytest.approx(0.5)

    def test_sidecar_default_threshold_is_zero(self, tmp_path: Path) -> None:
        """Default mask_threshold=0.0 is written to sidecar."""
        tsv_path = tmp_path / "parc.tsv"
        tsv_path.write_text("index\tlabel\n1\tregion1\n")

        write_parcellation_sidecar(
            tsv_path=tsv_path,
            original_file=tmp_path / "scalar.nii.gz",
            atlas_name="TestAtlas",
            atlas_image=tmp_path / "atlas.nii.gz",
            atlas_lut=None,
            mask=None,
        )

        sidecar = json.loads(tsv_path.with_suffix(".json").read_text())
        assert sidecar["mask_threshold"] == 0.0

    def test_sidecar_threshold_provenance_survives_round_trip(self, tmp_path: Path) -> None:
        """Threshold value is preserved exactly through JSON serialisation."""
        tsv_path = tmp_path / "parc.tsv"
        tsv_path.write_text("index\tlabel\n")

        threshold = 0.333
        write_parcellation_sidecar(
            tsv_path=tsv_path,
            original_file=tmp_path / "s.nii.gz",
            atlas_name="A",
            atlas_image=tmp_path / "a.nii.gz",
            atlas_lut=None,
            mask=None,
            mask_threshold=threshold,
        )
        sidecar = json.loads(tsv_path.with_suffix(".json").read_text())
        assert sidecar["mask_threshold"] == pytest.approx(threshold)


# ---------------------------------------------------------------------------
# CSV CLI env var
# ---------------------------------------------------------------------------


class TestMaskThresholdEnvVar:
    """Test MASKING_THRESHOLD env var is picked up by config_from_env."""

    def test_masking_threshold_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """MASKING_THRESHOLD env var is read into Cat12Config.mask_threshold."""
        from parcellate.interfaces.cat12.cli import config_from_env

        monkeypatch.setenv("CAT12_ROOT", str(tmp_path))
        monkeypatch.setenv("MASKING_THRESHOLD", "0.5")
        monkeypatch.delenv("CAT12_MASK", raising=False)
        monkeypatch.delenv("CAT12_OUTPUT_DIR", raising=False)
        monkeypatch.delenv("CAT12_ATLAS_PATHS", raising=False)
        monkeypatch.delenv("CAT12_LOG_LEVEL", raising=False)

        config = config_from_env()
        assert config.mask_threshold == pytest.approx(0.5)

    def test_missing_env_var_defaults_to_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When MASKING_THRESHOLD is not set, default is 0.0."""
        from parcellate.interfaces.cat12.cli import config_from_env

        monkeypatch.setenv("CAT12_ROOT", str(tmp_path))
        monkeypatch.delenv("MASKING_THRESHOLD", raising=False)
        monkeypatch.delenv("CAT12_MASK", raising=False)
        monkeypatch.delenv("CAT12_OUTPUT_DIR", raising=False)
        monkeypatch.delenv("CAT12_ATLAS_PATHS", raising=False)
        monkeypatch.delenv("CAT12_LOG_LEVEL", raising=False)

        config = config_from_env()
        assert config.mask_threshold == 0.0

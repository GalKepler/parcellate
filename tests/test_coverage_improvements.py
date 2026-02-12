"""Additional tests to improve code coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd

from parcellate.interfaces.cat12.cat12 import (
    build_arg_parser,
    load_config,
    run_parcellations,
)
from parcellate.interfaces.cat12.cat12 import (
    main as cat12_main,
)


class TestCAT12MainCLI:
    """Tests for CAT12 main CLI entry point."""

    def test_main_with_valid_config(self, tmp_path: Path) -> None:
        """Test main() with valid configuration."""
        # Create minimal config
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
input_root = "{tmp_path / "input"}"
output_dir = "{tmp_path / "output"}"

[[atlases]]
name = "test_atlas"
path = "{tmp_path / "atlas.nii.gz"}"
space = "MNI152NLin2009cAsym"
"""
        )

        with patch("parcellate.interfaces.cat12.cat12.run_parcellations") as mock_run:
            mock_run.return_value = []
            result = cat12_main([str(config_file)])

            assert result == 0
            mock_run.assert_called_once()

    def test_main_with_exception(self, tmp_path: Path) -> None:
        """Test main() when run_parcellations raises an exception."""
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
input_root = "{tmp_path / "input"}"
output_dir = "{tmp_path / "output"}"

[[atlases]]
name = "test_atlas"
path = "{tmp_path / "atlas.nii.gz"}"
space = "MNI152NLin2009cAsym"
"""
        )

        with patch("parcellate.interfaces.cat12.cat12.run_parcellations") as mock_run:
            mock_run.side_effect = RuntimeError("Test error")
            result = cat12_main([str(config_file)])

            assert result == 1

    def test_build_arg_parser(self) -> None:
        """Test argument parser creation."""
        parser = build_arg_parser()
        assert parser is not None

        # Test that it accepts config argument
        args = parser.parse_args(["config.toml"])
        assert args.config == Path("config.toml")


class TestCAT12ParallelProcessing:
    """Tests for CAT12 parallel processing."""

    def test_run_parcellations_with_parallel_processing(self, tmp_path: Path) -> None:
        """Test run_parcellations with n_procs > 1."""
        # Create fake input structure
        input_root = tmp_path / "input"
        subject_dir = input_root / "sub-01" / "anat"
        subject_dir.mkdir(parents=True)

        # Create fake GM map
        gm_map = subject_dir / "mwp1sub-01_T1w.nii.gz"
        data = np.random.rand(10, 10, 10).astype(np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), gm_map)

        # Create fake atlas
        atlas_path = tmp_path / "atlas.nii.gz"
        atlas_data = np.zeros((10, 10, 10), dtype=np.int16)
        atlas_data[2:4, 2:4, 2:4] = 1
        nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_path)

        # Create LUT
        lut_path = tmp_path / "atlas_lut.tsv"
        lut = pd.DataFrame({"index": [1], "label": ["Region1"]})
        lut.to_csv(lut_path, sep="\t", index=False)

        # Create config
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
input_root = "{input_root}"
output_dir = "{tmp_path / "output"}"
n_procs = 2

[[atlases]]
name = "test_atlas"
path = "{atlas_path}"
lut = "{lut_path}"
space = "MNI152NLin2009cAsym"
"""
        )

        config = load_config(config_file)
        outputs = run_parcellations(config)

        # Should have created output
        assert len(outputs) > 0
        assert all(p.exists() for p in outputs)

    def test_run_parcellations_with_no_inputs(self, tmp_path: Path) -> None:
        """Test run_parcellations when no inputs are discovered."""
        # Empty input directory
        input_root = tmp_path / "input"
        input_root.mkdir()

        # Create fake atlas
        atlas_path = tmp_path / "atlas.nii.gz"
        atlas_data = np.zeros((10, 10, 10), dtype=np.int16)
        atlas_data[2:4, 2:4, 2:4] = 1
        nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_path)

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
input_root = "{input_root}"
output_dir = "{tmp_path / "output"}"

[[atlases]]
name = "test_atlas"
path = "{atlas_path}"
space = "MNI152NLin2009cAsym"
"""
        )

        config = load_config(config_file)
        outputs = run_parcellations(config)

        # Should return empty list
        assert len(outputs) == 0


class TestParcellatorEdgeCases:
    """Tests for VolumetricParcellator edge cases."""

    def test_parcellator_with_negative_background_label(self, tmp_path: Path) -> None:
        """Test parcellator with negative background label."""
        from parcellate.parcellation.volume import VolumetricParcellator

        # Create atlas with negative background
        atlas_data = np.full((10, 10, 10), -1, dtype=np.int16)
        atlas_data[2:4, 2:4, 2:4] = 1
        atlas_data[6:8, 6:8, 6:8] = 2

        atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))
        lut = pd.DataFrame({"index": [1, 2], "label": ["R1", "R2"]})

        vp = VolumetricParcellator(atlas_img=atlas_img, lut=lut, background_label=-1)

        # Create scalar map
        scalar_data = np.random.rand(10, 10, 10).astype(np.float32)
        scalar_img = nib.Nifti1Image(scalar_data, np.eye(4))

        vp.fit(scalar_img)
        result = vp.transform(scalar_img)

        # Should only have regions 1 and 2
        assert len(result) == 2
        assert set(result["index"]) == {1, 2}

    def test_parcellator_resampling_target_atlas(self, tmp_path: Path) -> None:
        """Test parcellator with resampling_target='atlas'."""
        from parcellate.parcellation.volume import VolumetricParcellator

        atlas_data = np.zeros((10, 10, 10), dtype=np.int16)
        atlas_data[2:4, 2:4, 2:4] = 1

        atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))
        lut = pd.DataFrame({"index": [1], "label": ["R1"]})

        vp = VolumetricParcellator(atlas_img=atlas_img, lut=lut, resampling_target="atlas")

        scalar_data = np.random.rand(10, 10, 10).astype(np.float32)
        scalar_img = nib.Nifti1Image(scalar_data, np.eye(4))

        vp.fit(scalar_img)
        result = vp.transform(scalar_img)

        assert len(result) == 1
        assert result["index"].iloc[0] == 1

    def test_parcellator_with_mask_caching(self, tmp_path: Path) -> None:
        """Test that mask and atlas caching works correctly."""
        from parcellate.parcellation.volume import VolumetricParcellator

        atlas_data = np.zeros((10, 10, 10), dtype=np.int16)
        atlas_data[2:8, 2:8, 2:8] = 1

        atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))
        lut = pd.DataFrame({"index": [1], "label": ["R1"]})

        # Create mask
        mask_data = np.ones((10, 10, 10), dtype=np.uint8)
        mask_data[5:, :, :] = 0  # Mask out half the volume
        mask_img = nib.Nifti1Image(mask_data, np.eye(4))

        vp = VolumetricParcellator(atlas_img=atlas_img, lut=lut, mask=mask_img)

        # Create two scalar maps with same dimensions
        scalar_data1 = np.random.rand(10, 10, 10).astype(np.float32)
        scalar_img1 = nib.Nifti1Image(scalar_data1, np.eye(4))

        scalar_data2 = np.random.rand(10, 10, 10).astype(np.float32)
        scalar_img2 = nib.Nifti1Image(scalar_data2, np.eye(4))

        # Fit with first image
        vp.fit(scalar_img1)
        result1 = vp.transform(scalar_img1)

        # Fit with second image (should use cached mask/atlas)
        vp.fit(scalar_img2)
        result2 = vp.transform(scalar_img2)

        # Both should have same structure
        assert len(result1) == 1
        assert len(result2) == 1
        assert result1["index"].iloc[0] == 1
        assert result2["index"].iloc[0] == 1

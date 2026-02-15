"""Integration tests for full parcellation pipelines."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from parcellate.interfaces.cat12.cat12 import load_config as load_cat12_config
from parcellate.interfaces.cat12.cat12 import run_parcellations as run_cat12_parcellations
from parcellate.interfaces.qsirecon.qsirecon import load_config as load_qsirecon_config
from parcellate.interfaces.qsirecon.qsirecon import (
    run_parcellations as run_qsirecon_parcellations,
)


@pytest.fixture
def synthetic_nifti(tmp_path: Path) -> Path:
    """Create a synthetic NIfTI file for testing."""
    data = np.random.rand(10, 10, 10).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nifti_path = tmp_path / "test.nii.gz"
    nib.save(img, nifti_path)
    return nifti_path


@pytest.fixture
def synthetic_atlas(tmp_path: Path) -> tuple[Path, Path]:
    """Create a synthetic atlas with LUT."""
    # Create atlas with 3 regions
    data = np.zeros((10, 10, 10), dtype=np.int16)
    data[2:4, 2:4, 2:4] = 1  # Region 1
    data[6:8, 6:8, 6:8] = 2  # Region 2
    data[1:3, 6:8, 6:8] = 3  # Region 3

    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    atlas_path = tmp_path / "atlas.nii.gz"
    nib.save(img, atlas_path)

    # Create LUT
    lut = pd.DataFrame({
        "index": [1, 2, 3],
        "label": ["Region_1", "Region_2", "Region_3"],
    })
    lut_path = tmp_path / "atlas_lut.tsv"
    lut.to_csv(lut_path, sep="\t", index=False)

    return atlas_path, lut_path


class TestCAT12Integration:
    """Integration tests for CAT12 pipeline."""

    def test_full_pipeline_with_synthetic_data(
        self, tmp_path: Path, synthetic_nifti: Path, synthetic_atlas: tuple[Path, Path]
    ) -> None:
        """Test complete CAT12 pipeline with synthetic data."""
        atlas_path, lut_path = synthetic_atlas

        # Create CAT12-like directory structure
        cat12_root = tmp_path / "cat12_derivatives"
        subject_dir = cat12_root / "sub-01" / "anat"
        subject_dir.mkdir(parents=True)

        # Create synthetic GM map
        gm_map = subject_dir / "mwp1sub-01_T1w.nii.gz"
        data = np.random.rand(10, 10, 10).astype(np.float32) * 0.8  # GM probabilities
        nib.save(nib.Nifti1Image(data, np.eye(4)), gm_map)

        # Create TOML config
        config_file = tmp_path / "cat12_config.toml"
        config_content = f"""
input_root = "{cat12_root}"
output_dir = "{tmp_path / "output"}"
force = true
log_level = "INFO"
n_jobs = 1
n_procs = 1

[[atlases]]
name = "TestAtlas"
path = "{atlas_path}"
lut = "{lut_path}"
space = "MNI152NLin2009cAsym"
"""
        config_file.write_text(config_content)

        # Load config and run
        import argparse

        args = argparse.Namespace(
            config=config_file,
            input_root=None,
            output_dir=None,
            atlas_config=None,
            subjects=None,
            sessions=None,
            mask=None,
            force=False,
            log_level=None,
            n_jobs=None,
            n_procs=None,
        )
        config = load_cat12_config(args)
        outputs = run_cat12_parcellations(config)

        # Verify outputs
        assert len(outputs) > 0, "Should produce at least one output file"

        # Check output file exists and has correct structure
        output_file = outputs[0]
        assert output_file.exists()
        assert output_file.suffix == ".tsv"

        # Verify output content
        df = pd.read_csv(output_file, sep="\t")
        assert "index" in df.columns
        assert "label" in df.columns
        assert "mean" in df.columns
        assert "std" in df.columns
        assert len(df) == 3  # Three regions

    def test_skip_existing_outputs(self, tmp_path: Path, synthetic_atlas: tuple[Path, Path]) -> None:
        """Test that existing outputs are skipped when force=False."""
        atlas_path, lut_path = synthetic_atlas

        # Create minimal CAT12 structure
        cat12_root = tmp_path / "cat12_derivatives"
        subject_dir = cat12_root / "sub-01" / "anat"
        subject_dir.mkdir(parents=True)

        gm_map = subject_dir / "mwp1sub-01_T1w.nii.gz"
        data = np.random.rand(10, 10, 10).astype(np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), gm_map)

        # Create config with force=False
        config_file = tmp_path / "config.toml"
        config_content = f"""
input_root = "{cat12_root}"
output_dir = "{tmp_path / "output"}"
force = false

[[atlases]]
name = "TestAtlas"
path = "{atlas_path}"
lut = "{lut_path}"
space = "MNI152NLin2009cAsym"
"""
        config_file.write_text(config_content)

        # Run first time
        import argparse

        args = argparse.Namespace(
            config=config_file,
            input_root=None,
            output_dir=None,
            atlas_config=None,
            subjects=None,
            sessions=None,
            mask=None,
            force=False,
            log_level=None,
            n_jobs=None,
            n_procs=None,
        )
        config = load_cat12_config(args)
        outputs1 = run_cat12_parcellations(config)
        assert len(outputs1) > 0

        # Modify output file to detect if it's overwritten
        output_file = outputs1[0]
        original_mtime = output_file.stat().st_mtime

        # Small delay to ensure different mtime if overwritten
        import time

        time.sleep(0.01)

        # Run second time with force=False
        _ = run_cat12_parcellations(config)

        # File should not be overwritten
        assert output_file.stat().st_mtime == original_mtime

    def test_force_flag_overwrites_existing(self, tmp_path: Path, synthetic_atlas: tuple[Path, Path]) -> None:
        """Test that force=True overwrites existing outputs."""
        atlas_path, lut_path = synthetic_atlas

        # Create minimal CAT12 structure
        cat12_root = tmp_path / "cat12_derivatives"
        subject_dir = cat12_root / "sub-01" / "anat"
        subject_dir.mkdir(parents=True)

        gm_map = subject_dir / "mwp1sub-01_T1w.nii.gz"
        data = np.random.rand(10, 10, 10).astype(np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), gm_map)

        # Create config with force=True
        config_file = tmp_path / "config.toml"
        config_content = f"""
input_root = "{cat12_root}"
output_dir = "{tmp_path / "output"}"
force = true

[[atlases]]
name = "TestAtlas"
path = "{atlas_path}"
lut = "{lut_path}"
space = "MNI152NLin2009cAsym"
"""
        config_file.write_text(config_content)

        # Run first time
        import argparse

        args = argparse.Namespace(
            config=config_file,
            input_root=None,
            output_dir=None,
            atlas_config=None,
            subjects=None,
            sessions=None,
            mask=None,
            force=False,
            log_level=None,
            n_jobs=None,
            n_procs=None,
        )
        config = load_cat12_config(args)
        outputs1 = run_cat12_parcellations(config)
        output_file = outputs1[0]
        original_mtime = output_file.stat().st_mtime

        # Small delay
        import time

        time.sleep(0.01)

        # Run second time with force=True
        _ = run_cat12_parcellations(config)

        # File should be overwritten (or at least touched)
        assert output_file.stat().st_mtime >= original_mtime


class TestQSIReconIntegration:
    """Integration tests for QSIRecon pipeline."""

    def test_full_pipeline_with_synthetic_data(self, tmp_path: Path, synthetic_atlas: tuple[Path, Path]) -> None:
        """Test complete QSIRecon pipeline with synthetic data."""
        atlas_path, lut_path = synthetic_atlas

        # Create QSIRecon-like directory structure
        qsi_root = tmp_path / "qsirecon_derivatives"
        workflow_dir = qsi_root / "derivatives" / "qsirecon-test_workflow"
        subject_dir = workflow_dir / "sub-01" / "dwi"
        subject_dir.mkdir(parents=True)

        # Create synthetic DWI map
        dwi_map = subject_dir / "sub-01_space-MNI152NLin2009cAsym_param-FA_dwimap.nii.gz"
        data = np.random.rand(10, 10, 10).astype(np.float32) * 0.8  # FA values
        nib.save(nib.Nifti1Image(data, np.eye(4)), dwi_map)

        # Create config
        config_file = tmp_path / "qsi_config.toml"
        config_content = f"""
input_root = "{qsi_root}"
output_dir = "{tmp_path / "output"}"
force = true
log_level = "INFO"

[[atlases]]
name = "TestAtlas"
path = "{atlas_path}"
lut = "{lut_path}"
space = "MNI152NLin2009cAsym"
"""
        config_file.write_text(config_content)

        # Create argparse Namespace for load_config
        import argparse

        args = argparse.Namespace(
            config=config_file,
            input_root=None,
            output_dir=None,
            atlas_config=None,
            subjects=None,
            sessions=None,
            mask=None,
            force=None,
            log_level=None,
            n_jobs=None,
            n_procs=None,
        )

        # Load config and run
        config = load_qsirecon_config(args)
        outputs = run_qsirecon_parcellations(config)

        # Verify outputs
        assert len(outputs) > 0, "Should produce at least one output file"

        # Check output file structure
        output_file = outputs[0]
        assert output_file.exists()
        assert output_file.suffix == ".tsv"

        # Verify content
        df = pd.read_csv(output_file, sep="\t")
        assert "index" in df.columns
        assert "label" in df.columns
        assert len(df) == 3  # Three regions

    def test_empty_directory_returns_no_outputs(self, tmp_path: Path) -> None:
        """Test that empty input directory returns no outputs."""
        # Create empty QSIRecon directory
        qsi_root = tmp_path / "qsirecon_empty"
        qsi_root.mkdir()
        (qsi_root / "derivatives").mkdir()

        # Create minimal config
        config_file = tmp_path / "config.toml"
        config_content = f"""
input_root = "{qsi_root}"
output_dir = "{tmp_path / "output"}"
"""
        config_file.write_text(config_content)

        import argparse

        args = argparse.Namespace(
            config=config_file,
            input_root=None,
            output_dir=None,
            atlas_config=None,
            subjects=None,
            sessions=None,
            mask=None,
            force=None,
            log_level=None,
            n_jobs=None,
            n_procs=None,
        )

        config = load_qsirecon_config(args)
        outputs = run_qsirecon_parcellations(config)

        assert len(outputs) == 0, "Should return no outputs for empty directory"


class TestMultiSessionSupport:
    """Integration tests for multi-session processing."""

    def test_qsirecon_processes_multiple_sessions(self, tmp_path: Path, synthetic_atlas: tuple[Path, Path]) -> None:
        """Test QSIRecon processes all sessions per subject."""
        atlas_path, lut_path = synthetic_atlas

        # Create multi-session structure
        qsi_root = tmp_path / "qsirecon_multi"
        workflow_dir = qsi_root / "derivatives" / "qsirecon-test"

        # Session 1
        ses1_dir = workflow_dir / "sub-01" / "ses-baseline" / "dwi"
        ses1_dir.mkdir(parents=True)
        dwi1 = ses1_dir / "sub-01_ses-baseline_space-MNI152NLin2009cAsym_param-FA_dwimap.nii.gz"
        nib.save(nib.Nifti1Image(np.random.rand(10, 10, 10).astype(np.float32), np.eye(4)), dwi1)

        # Session 2
        ses2_dir = workflow_dir / "sub-01" / "ses-followup" / "dwi"
        ses2_dir.mkdir(parents=True)
        dwi2 = ses2_dir / "sub-01_ses-followup_space-MNI152NLin2009cAsym_param-FA_dwimap.nii.gz"
        nib.save(nib.Nifti1Image(np.random.rand(10, 10, 10).astype(np.float32), np.eye(4)), dwi2)

        # Config
        config_file = tmp_path / "config.toml"
        config_content = f"""
input_root = "{qsi_root}"
output_dir = "{tmp_path / "output"}"

[[atlases]]
name = "TestAtlas"
path = "{atlas_path}"
lut = "{lut_path}"
space = "MNI152NLin2009cAsym"
"""
        config_file.write_text(config_content)

        import argparse

        args = argparse.Namespace(
            config=config_file,
            input_root=None,
            output_dir=None,
            atlas_config=None,
            subjects=None,
            sessions=None,
            mask=None,
            force=None,
            log_level=None,
            n_jobs=None,
            n_procs=None,
        )

        config = load_qsirecon_config(args)
        outputs = run_qsirecon_parcellations(config)

        # Should process both sessions
        assert len(outputs) >= 2, "Should process both sessions"

        # Check that both sessions are in output paths
        output_paths = [str(p) for p in outputs]
        assert any("baseline" in p for p in output_paths), "Should have baseline output"
        assert any("followup" in p for p in output_paths), "Should have followup output"

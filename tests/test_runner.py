"""Tests for parcellation workflow runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from parcellate.interfaces.models import (
    AtlasDefinition,
    ParcellationConfig,
    ReconInput,
    SubjectContext,
)
from parcellate.interfaces.runner import (
    ScalarMapSpaceMismatchError,
    _validate_scalar_map_spaces,
    run_parcellation_workflow,
)


class MockScalarMap:
    """Mock scalar map for testing."""

    def __init__(self, name: str, space: str | None = None):
        self.name = name
        self.space = space
        self.nifti_path = Path(f"/fake/{name}.nii.gz")
        self.desc = None


class TestValidateScalarMapSpaces:
    """Tests for scalar map space validation."""

    def test_validate_consistent_spaces(self) -> None:
        """Test that validation passes for consistent spaces."""
        scalar_maps = [
            MockScalarMap("map1", space="MNI152NLin2009cAsym"),
            MockScalarMap("map2", space="MNI152NLin2009cAsym"),
        ]

        # Should not raise
        _validate_scalar_map_spaces(scalar_maps)

    def test_validate_inconsistent_spaces_raises(self) -> None:
        """Test that validation raises for inconsistent spaces."""
        scalar_maps = [
            MockScalarMap("map1", space="MNI152NLin2009cAsym"),
            MockScalarMap("map2", space="T1w"),
        ]

        with pytest.raises(ScalarMapSpaceMismatchError) as exc_info:
            _validate_scalar_map_spaces(scalar_maps)

        assert "inconsistent" in str(exc_info.value).lower()
        assert "MNI152NLin2009cAsym" in str(exc_info.value)
        assert "T1w" in str(exc_info.value)

    def test_validate_empty_list(self) -> None:
        """Test that validation handles empty list."""
        scalar_maps = []

        # Should not raise
        _validate_scalar_map_spaces(scalar_maps)

    def test_validate_single_map(self) -> None:
        """Test that validation passes for single map."""
        scalar_maps = [MockScalarMap("map1", space="MNI152NLin2009cAsym")]

        # Should not raise
        _validate_scalar_map_spaces(scalar_maps)

    def test_validate_none_spaces_consistent(self) -> None:
        """Test that None spaces are treated consistently."""
        scalar_maps = [
            MockScalarMap("map1", space=None),
            MockScalarMap("map2", space=None),
        ]

        # Should not raise (all have None space)
        _validate_scalar_map_spaces(scalar_maps)

    def test_validate_mixed_none_and_space_raises(self) -> None:
        """Test that mixing None and non-None spaces raises."""
        scalar_maps = [
            MockScalarMap("map1", space=None),
            MockScalarMap("map2", space="MNI152NLin2009cAsym"),
        ]

        with pytest.raises(ScalarMapSpaceMismatchError):
            _validate_scalar_map_spaces(scalar_maps)


class TestRunParcellationWorkflow:
    """Tests for workflow runner."""

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> ParcellationConfig:
        """Create mock configuration."""
        return ParcellationConfig(
            input_root=tmp_path / "input",
            output_dir=tmp_path / "output",
            mask=None,
            background_label=0,
            resampling_target="data",
            n_jobs=1,
        )

    @pytest.fixture
    def mock_recon(self) -> ReconInput:
        """Create mock reconstruction input."""
        context = SubjectContext(subject_id="01", session_id=None)
        atlases = [
            AtlasDefinition(
                name="atlas1",
                nifti_path=Path("/fake/atlas1.nii.gz"),
                space="MNI152NLin2009cAsym",
            ),
        ]
        scalar_maps = [MockScalarMap("map1", space="MNI152NLin2009cAsym")]
        return ReconInput(context=context, atlases=atlases, scalar_maps=scalar_maps)

    def test_run_workflow_single_atlas(self, mock_recon: ReconInput, mock_config: ParcellationConfig) -> None:
        """Test running workflow with single atlas."""
        plan = {mock_recon.atlases[0]: [MockScalarMap("map1", space="MNI152NLin2009cAsym")]}

        mock_stats = pd.DataFrame({
            "index": [1, 2],
            "label": ["Region1", "Region2"],
            "mean": [0.5, 0.6],
        })

        with patch("parcellate.interfaces.runner.VolumetricParcellator") as mock_vp_class:
            mock_vp = MagicMock()
            mock_vp.transform.return_value = mock_stats
            mock_vp_class.return_value = mock_vp

            outputs = run_parcellation_workflow(mock_recon, plan, mock_config)

            assert len(outputs) == 1
            assert outputs[0].context == mock_recon.context
            assert outputs[0].atlas == mock_recon.atlases[0]
            mock_vp.fit.assert_called_once()
            mock_vp.transform.assert_called_once()

    def test_run_workflow_empty_plan(self, mock_recon: ReconInput, mock_config: ParcellationConfig) -> None:
        """Test running workflow with empty plan."""
        plan = {}

        outputs = run_parcellation_workflow(mock_recon, plan, mock_config)

        assert len(outputs) == 0

    def test_run_workflow_skips_empty_scalar_maps(
        self, mock_recon: ReconInput, mock_config: ParcellationConfig
    ) -> None:
        """Test that workflow skips atlases with no scalar maps."""
        plan = {mock_recon.atlases[0]: []}  # Empty list

        outputs = run_parcellation_workflow(mock_recon, plan, mock_config)

        assert len(outputs) == 0

    def test_run_workflow_continues_after_space_mismatch(
        self, mock_recon: ReconInput, mock_config: ParcellationConfig
    ) -> None:
        """Test that workflow continues after encountering space mismatch."""
        atlas1 = AtlasDefinition(
            name="atlas1",
            nifti_path=Path("/fake/atlas1.nii.gz"),
            space="MNI152NLin2009cAsym",
        )
        atlas2 = AtlasDefinition(
            name="atlas2",
            nifti_path=Path("/fake/atlas2.nii.gz"),
            space="T1w",
        )

        # First atlas has mismatched scalar maps (will be skipped)
        # Second atlas has valid scalar maps
        plan = {
            atlas1: [
                MockScalarMap("map1", space="MNI152NLin2009cAsym"),
                MockScalarMap("map2", space="T1w"),  # Mismatch!
            ],
            atlas2: [MockScalarMap("map3", space="T1w")],
        }

        mock_stats = pd.DataFrame({
            "index": [1],
            "label": ["Region1"],
            "mean": [0.5],
        })

        with patch("parcellate.interfaces.runner.VolumetricParcellator") as mock_vp_class:
            mock_vp = MagicMock()
            mock_vp.transform.return_value = mock_stats
            mock_vp_class.return_value = mock_vp

            outputs = run_parcellation_workflow(mock_recon, plan, mock_config)

            # Should only have output from atlas2 (atlas1 skipped due to mismatch)
            assert len(outputs) == 1
            assert outputs[0].atlas == atlas2

    def test_run_workflow_continues_after_parcellator_init_error(
        self, mock_recon: ReconInput, mock_config: ParcellationConfig
    ) -> None:
        """Test that workflow continues after parcellator initialization fails."""
        atlas1 = AtlasDefinition(
            name="atlas1",
            nifti_path=Path("/fake/atlas1.nii.gz"),
            space="MNI152NLin2009cAsym",
        )
        atlas2 = AtlasDefinition(
            name="atlas2",
            nifti_path=Path("/fake/atlas2.nii.gz"),
            space="MNI152NLin2009cAsym",
        )

        plan = {
            atlas1: [MockScalarMap("map1", space="MNI152NLin2009cAsym")],
            atlas2: [MockScalarMap("map2", space="MNI152NLin2009cAsym")],
        }

        mock_stats = pd.DataFrame({
            "index": [1],
            "label": ["Region1"],
            "mean": [0.5],
        })

        with patch("parcellate.interfaces.runner.VolumetricParcellator") as mock_vp_class:
            # First call raises error, second succeeds
            mock_vp_success = MagicMock()
            mock_vp_success.transform.return_value = mock_stats

            def side_effect(*args, **kwargs):
                if mock_vp_class.call_count == 1:
                    raise ValueError("Init failed")  # noqa: TRY003
                return mock_vp_success

            mock_vp_class.side_effect = side_effect

            outputs = run_parcellation_workflow(mock_recon, plan, mock_config)

            # Should only have output from atlas2
            assert len(outputs) == 1
            assert outputs[0].atlas == atlas2

    def test_run_workflow_continues_after_transform_error(
        self, mock_recon: ReconInput, mock_config: ParcellationConfig
    ) -> None:
        """Test that workflow continues after transform fails."""
        plan = {
            mock_recon.atlases[0]: [
                MockScalarMap("map1", space="MNI152NLin2009cAsym"),
                MockScalarMap("map2", space="MNI152NLin2009cAsym"),
            ]
        }

        mock_stats = pd.DataFrame({
            "index": [1],
            "label": ["Region1"],
            "mean": [0.5],
        })

        with patch("parcellate.interfaces.runner.VolumetricParcellator") as mock_vp_class:
            mock_vp = MagicMock()
            # First transform fails, second succeeds
            mock_vp.transform.side_effect = [
                ValueError("Transform failed"),
                mock_stats,
            ]
            mock_vp_class.return_value = mock_vp

            outputs = run_parcellation_workflow(mock_recon, plan, mock_config)

            # Should only have one output (second map succeeded)
            assert len(outputs) == 1
            assert outputs[0].scalar_map.name == "map2"

    def test_run_workflow_multiple_scalar_maps(self, mock_recon: ReconInput, mock_config: ParcellationConfig) -> None:
        """Test running workflow with multiple scalar maps."""
        plan = {
            mock_recon.atlases[0]: [
                MockScalarMap("map1", space="MNI152NLin2009cAsym"),
                MockScalarMap("map2", space="MNI152NLin2009cAsym"),
                MockScalarMap("map3", space="MNI152NLin2009cAsym"),
            ]
        }

        mock_stats = pd.DataFrame({
            "index": [1],
            "label": ["Region1"],
            "mean": [0.5],
        })

        with patch("parcellate.interfaces.runner.VolumetricParcellator") as mock_vp_class:
            mock_vp = MagicMock()
            mock_vp.transform.return_value = mock_stats
            mock_vp_class.return_value = mock_vp

            outputs = run_parcellation_workflow(mock_recon, plan, mock_config)

            assert len(outputs) == 3
            assert mock_vp.transform.call_count == 3
            assert {o.scalar_map.name for o in outputs} == {"map1", "map2", "map3"}

    def test_run_workflow_uses_config_settings(self, mock_recon: ReconInput, tmp_path: Path) -> None:
        """Test that workflow uses configuration settings."""
        mask_path = tmp_path / "mask.nii.gz"
        config = ParcellationConfig(
            input_root=tmp_path / "input",
            output_dir=tmp_path / "output",
            mask=mask_path,
            background_label=255,
            resampling_target="labels",
            n_jobs=4,
        )

        plan = {mock_recon.atlases[0]: [MockScalarMap("map1", space="MNI152NLin2009cAsym")]}

        mock_stats = pd.DataFrame({
            "index": [1],
            "label": ["Region1"],
            "mean": [0.5],
        })

        with patch("parcellate.interfaces.runner.VolumetricParcellator") as mock_vp_class:
            mock_vp = MagicMock()
            mock_vp.transform.return_value = mock_stats
            mock_vp_class.return_value = mock_vp

            outputs = run_parcellation_workflow(mock_recon, plan, config)

            # Verify VolumetricParcellator was called with config settings
            mock_vp_class.assert_called_once()
            call_kwargs = mock_vp_class.call_args.kwargs
            assert call_kwargs["mask"] == mask_path
            assert call_kwargs["background_label"] == 255
            assert call_kwargs["resampling_target"] == "labels"

            assert len(outputs) == 1

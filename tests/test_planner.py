"""Tests for parcellation workflow planning."""

from __future__ import annotations

from pathlib import Path

from parcellate.interfaces.models import AtlasDefinition, ReconInput, SubjectContext
from parcellate.interfaces.planner import _space_match, plan_parcellation_workflow


class MockScalarMap:
    """Mock scalar map for testing."""

    def __init__(self, name: str, space: str | None = None):
        self.name = name
        self.space = space
        self.nifti_path = Path(f"/fake/{name}.nii.gz")


class TestSpaceMatch:
    """Tests for space matching utility function."""

    def test_matching_spaces_returns_true(self) -> None:
        """Test that matching spaces return True."""
        atlas = AtlasDefinition(name="atlas", nifti_path=Path("/fake/atlas.nii.gz"), space="MNI152NLin2009cAsym")
        scalar_map = MockScalarMap("map", space="MNI152NLin2009cAsym")
        assert _space_match(atlas, scalar_map) is True

    def test_case_insensitive_match(self) -> None:
        """Test that space matching is case-insensitive."""
        atlas = AtlasDefinition(name="atlas", nifti_path=Path("/fake/atlas.nii.gz"), space="MNI152NLin2009cAsym")
        scalar_map = MockScalarMap("map", space="mni152nlin2009casym")
        assert _space_match(atlas, scalar_map) is True

    def test_different_spaces_returns_false(self) -> None:
        """Test that different spaces return False."""
        atlas = AtlasDefinition(name="atlas", nifti_path=Path("/fake/atlas.nii.gz"), space="MNI152NLin2009cAsym")
        scalar_map = MockScalarMap("map", space="MNI152NLin6Asym")
        assert _space_match(atlas, scalar_map) is False

    def test_none_values_return_false(self) -> None:
        """Test that None values return False."""
        atlas_with_space = AtlasDefinition(
            name="atlas", nifti_path=Path("/fake/atlas.nii.gz"), space="MNI152NLin2009cAsym"
        )
        atlas_without_space = AtlasDefinition(name="atlas", nifti_path=Path("/fake/atlas.nii.gz"), space=None)
        scalar_map_with_space = MockScalarMap("map", space="MNI152NLin2009cAsym")
        scalar_map_without_space = MockScalarMap("map", space=None)

        assert _space_match(atlas_without_space, scalar_map_with_space) is False
        assert _space_match(atlas_with_space, scalar_map_without_space) is False
        assert _space_match(atlas_without_space, scalar_map_without_space) is False

    def test_empty_strings_return_false(self) -> None:
        """Test that empty strings return False."""
        atlas = AtlasDefinition(name="atlas", nifti_path=Path("/fake/atlas.nii.gz"), space="")
        scalar_map = MockScalarMap("map", space="MNI152NLin2009cAsym")
        assert _space_match(atlas, scalar_map) is False

        atlas = AtlasDefinition(name="atlas", nifti_path=Path("/fake/atlas.nii.gz"), space="MNI152NLin2009cAsym")
        scalar_map = MockScalarMap("map", space="")
        assert _space_match(atlas, scalar_map) is False


class TestPlanParcellationWorkflow:
    """Tests for workflow planning function."""

    def test_plan_matches_by_space(self) -> None:
        """Test that planner matches atlases and scalar maps by space."""
        context = SubjectContext(subject_id="01", session_id=None)
        atlases = [
            AtlasDefinition(
                name="atlas1",
                nifti_path=Path("/fake/atlas1.nii.gz"),
                space="MNI152NLin2009cAsym",
            ),
        ]
        scalar_maps = [
            MockScalarMap("map1", space="MNI152NLin2009cAsym"),
            MockScalarMap("map2", space="T1w"),
        ]
        recon = ReconInput(context=context, atlases=atlases, scalar_maps=scalar_maps)

        plan = plan_parcellation_workflow(recon)

        # Should only match map1 to atlas1 (matching spaces)
        assert len(plan) == 1
        assert atlases[0] in plan
        assert len(plan[atlases[0]]) == 1
        assert plan[atlases[0]][0].name == "map1"

    def test_plan_no_match_different_spaces(self) -> None:
        """Test that planner returns empty list for atlas when no spaces match."""
        context = SubjectContext(subject_id="01", session_id=None)
        atlases = [
            AtlasDefinition(
                name="atlas1",
                nifti_path=Path("/fake/atlas1.nii.gz"),
                space="MNI152NLin2009cAsym",
            ),
        ]
        scalar_maps = [
            MockScalarMap("map1", space="T1w"),
            MockScalarMap("map2", space="fsaverage"),
        ]
        recon = ReconInput(context=context, atlases=atlases, scalar_maps=scalar_maps)

        plan = plan_parcellation_workflow(recon)

        # Atlas is in plan but has no matches
        assert len(plan) == 1
        assert atlases[0] in plan
        assert len(plan[atlases[0]]) == 0

    def test_plan_empty_atlases(self) -> None:
        """Test that planner handles empty atlas list."""
        context = SubjectContext(subject_id="01", session_id=None)
        atlases = []
        scalar_maps = [MockScalarMap("map1", space="MNI152NLin2009cAsym")]
        recon = ReconInput(context=context, atlases=atlases, scalar_maps=scalar_maps)

        plan = plan_parcellation_workflow(recon)

        assert len(plan) == 0

    def test_plan_empty_scalar_maps(self) -> None:
        """Test that planner handles empty scalar map list."""
        context = SubjectContext(subject_id="01", session_id=None)
        atlases = [
            AtlasDefinition(
                name="atlas1",
                nifti_path=Path("/fake/atlas1.nii.gz"),
                space="MNI152NLin2009cAsym",
            ),
        ]
        scalar_maps = []
        recon = ReconInput(context=context, atlases=atlases, scalar_maps=scalar_maps)

        plan = plan_parcellation_workflow(recon)

        # Atlas is in plan but has no matches
        assert len(plan) == 1
        assert atlases[0] in plan
        assert len(plan[atlases[0]]) == 0

    def test_plan_multiple_atlases_multiple_maps(self) -> None:
        """Test planning with multiple atlases and scalar maps."""
        context = SubjectContext(subject_id="01", session_id=None)
        atlases = [
            AtlasDefinition(
                name="atlas_mni",
                nifti_path=Path("/fake/atlas_mni.nii.gz"),
                space="MNI152NLin2009cAsym",
            ),
            AtlasDefinition(
                name="atlas_t1w",
                nifti_path=Path("/fake/atlas_t1w.nii.gz"),
                space="T1w",
            ),
        ]
        scalar_maps = [
            MockScalarMap("map_mni1", space="MNI152NLin2009cAsym"),
            MockScalarMap("map_mni2", space="MNI152NLin2009cAsym"),
            MockScalarMap("map_t1w", space="T1w"),
        ]
        recon = ReconInput(context=context, atlases=atlases, scalar_maps=scalar_maps)

        plan = plan_parcellation_workflow(recon)

        # atlas_mni should match map_mni1 and map_mni2
        assert atlases[0] in plan
        assert len(plan[atlases[0]]) == 2
        assert {m.name for m in plan[atlases[0]]} == {"map_mni1", "map_mni2"}

        # atlas_t1w should match map_t1w
        assert atlases[1] in plan
        assert len(plan[atlases[1]]) == 1
        assert plan[atlases[1]][0].name == "map_t1w"

    def test_plan_atlas_without_space(self) -> None:
        """Test that atlas without space doesn't match anything."""
        context = SubjectContext(subject_id="01", session_id=None)
        atlases = [
            AtlasDefinition(
                name="atlas1",
                nifti_path=Path("/fake/atlas1.nii.gz"),
                space=None,  # No space
            ),
        ]
        scalar_maps = [
            MockScalarMap("map1", space="MNI152NLin2009cAsym"),
        ]
        recon = ReconInput(context=context, atlases=atlases, scalar_maps=scalar_maps)

        plan = plan_parcellation_workflow(recon)

        # Atlas is in plan but has no matches
        assert len(plan) == 1
        assert atlases[0] in plan
        assert len(plan[atlases[0]]) == 0

    def test_plan_scalar_map_without_space(self) -> None:
        """Test that scalar map without space doesn't match anything."""
        context = SubjectContext(subject_id="01", session_id=None)
        atlases = [
            AtlasDefinition(
                name="atlas1",
                nifti_path=Path("/fake/atlas1.nii.gz"),
                space="MNI152NLin2009cAsym",
            ),
        ]
        scalar_maps = [
            MockScalarMap("map1", space=None),  # No space
        ]
        recon = ReconInput(context=context, atlases=atlases, scalar_maps=scalar_maps)

        plan = plan_parcellation_workflow(recon)

        # Atlas is in plan but has no matches
        assert len(plan) == 1
        assert atlases[0] in plan
        assert len(plan[atlases[0]]) == 0

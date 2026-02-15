from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from parcellate.interfaces.planner import _space_match
from parcellate.interfaces.planner import plan_parcellation_workflow as plan_qsirecon_parcellation_workflow
from parcellate.interfaces.qsirecon.loader import (
    discover_atlases,
    discover_scalar_maps,
    load_qsirecon_inputs,
)
from parcellate.interfaces.qsirecon.models import (
    AtlasDefinition,
    ParcellationOutput,
    QSIReconConfig,
    ScalarMapDefinition,
    SubjectContext,
)
from parcellate.interfaces.qsirecon.qsirecon import (
    _build_output_path,
    _write_output,
    load_config,
    run_parcellations,
)
from parcellate.interfaces.runner import run_parcellation_workflow as run_qsirecon_parcellation_workflow
from parcellate.interfaces.utils import _as_list, _parse_log_level


class Recon:
    def __init__(self, context, atlases, scalar_maps) -> None:
        self.context = context
        self.atlases = atlases
        self.scalar_maps = scalar_maps


class FakeFile:
    def __init__(self, path: Path, entities: dict[str, Any]) -> None:
        self.path = str(path)
        self.entities = entities
        self.filename = Path(path).name

    def get_entities(self) -> dict[str, Any]:
        return dict(self.entities)


class FakeLayout:
    def __init__(
        self,
        root: Path,
        files: list[FakeFile],
        entities: dict[str, Any] | None = None,
        subjects: list[str] | None = None,
        sessions: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self._files = files
        self._entities = entities or {}
        self._subjects = subjects or []
        self._sessions = sessions or []

    def get_entities(self) -> dict[str, Any]:
        return dict(self._entities)

    def get_subjects(self) -> list[str]:
        return list(self._subjects)

    def get_sessions(self, subject: str | None = None) -> list[str]:
        return list(self._sessions)

    def get(self, return_type: str = "object", **filters: Any) -> list[FakeFile]:
        results: list[FakeFile] = []
        for f in self._files:
            entities = f.get_entities()
            match = True
            for key, value in filters.items():
                if key == "extension":
                    continue
                if value is None:
                    continue
                candidate = entities.get(key)
                if isinstance(value, list):
                    if candidate not in value:
                        match = False
                        break
                else:
                    if candidate != value:
                        match = False
                        break
            if match:
                results.append(f)
        return results


def test_parse_log_level_handles_common_inputs() -> None:
    assert _parse_log_level(None) == logging.INFO
    assert _parse_log_level(logging.DEBUG) == logging.DEBUG
    assert _parse_log_level("debug") == logging.DEBUG
    assert _parse_log_level("INFO") == logging.INFO


def test_as_list_normalizes_inputs() -> None:
    assert _as_list(None) is None
    assert _as_list("one") == ["one"]
    assert _as_list(["one", "two"]) == ["one", "two"]


def test_load_config_reads_toml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "qsirecon.toml"
    cfg_path.write_text(
        "\n".join([
            'input_root = "~/data"',
            'output_dir = "outdir"',
            'subjects = ["01", "02"]',
            'sessions = ["baseline"]',
            'mask = "mask.nii.gz"',
            "force = true",
            'log_level = "debug"',
        ])
    )

    args = argparse.Namespace(
        config=cfg_path,
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
    config = load_config(args)

    assert config.input_root == Path("~/data").expanduser().resolve()
    assert config.output_dir == Path("outdir").expanduser().resolve()
    assert config.subjects == ["01", "02"]
    assert config.sessions == ["baseline"]
    assert config.mask == Path("mask.nii.gz").expanduser().resolve()
    assert config.force is True
    assert config.log_level == logging.DEBUG


def test_write_output_creates_bids_like_path(tmp_path: Path) -> None:
    context = SubjectContext(subject_id="01", session_id="02")
    atlas = AtlasDefinition(
        name="atlasA",
        nifti_path=tmp_path / "atlas.nii.gz",
        resolution="2mm",
        space="MNI",
    )
    scalar = ScalarMapDefinition(
        name="mapA",
        nifti_path=tmp_path / "map.nii.gz",
        param="odi",
        model="gqi",
        desc="smoothed",
        recon_workflow="workflowX",
        space="MNI",
    )
    stats = pd.DataFrame({"index": [1], "value": [3.14]})
    po = ParcellationOutput(context=context, atlas=atlas, scalar_map=scalar, stats_table=stats)
    config = QSIReconConfig(input_root=tmp_path, output_dir=tmp_path)

    out_path = _write_output(po, destination=tmp_path, config=config)

    expected_dir = tmp_path / "qsirecon-workflowX" / "sub-01" / "ses-02" / "dwi" / "atlas-atlasA"
    assert out_path.parent == expected_dir
    assert out_path.name.startswith("sub-01_ses-02_atlas-atlasA_space-MNI_res-2mm_model-gqi_param-odi_desc-smoothed")
    assert out_path.exists()
    written = pd.read_csv(out_path, sep="\t")
    assert written.equals(stats)

    # JSON sidecar should also be created
    json_path = out_path.with_suffix(".json")
    assert json_path.exists()


def test_run_parcellations_reuses_existing_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    context = SubjectContext("01")
    atlas = AtlasDefinition("atlas", nifti_path=tmp_path / "atlas.nii.gz")
    scalar = ScalarMapDefinition("map", nifti_path=tmp_path / "map.nii.gz", param="fa")
    recon = Recon(context, [atlas], [scalar])

    out_path = _build_output_path(context, atlas, scalar, tmp_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame({"index": [1], "value": [2.0]})
    existing.to_csv(out_path, sep="\t", index=False)

    monkeypatch.setattr(
        "parcellate.interfaces.qsirecon.qsirecon.load_qsirecon_inputs",
        lambda *args, **kwargs: [recon],
    )
    monkeypatch.setattr(
        "parcellate.interfaces.qsirecon.qsirecon.plan_parcellation_workflow",
        lambda recon: {atlas: [scalar]},
    )
    monkeypatch.setattr(
        "parcellate.interfaces.qsirecon.qsirecon.run_parcellation_workflow",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Should not run when outputs exist")),
    )

    config = QSIReconConfig(input_root=tmp_path, output_dir=tmp_path, force=False)

    outputs = run_parcellations(config)

    assert outputs == [out_path]
    assert pd.read_csv(out_path, sep="\t").equals(existing)


def test_run_parcellations_overwrites_when_forced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    context = SubjectContext("01")
    atlas = AtlasDefinition("atlas", nifti_path=tmp_path / "atlas.nii.gz")
    scalar = ScalarMapDefinition("map", nifti_path=tmp_path / "map.nii.gz", param="fa")
    recon = Recon(context, [atlas], [scalar])

    out_path = _build_output_path(context, atlas, scalar, tmp_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": [1], "value": [2.0]}).to_csv(out_path, sep="\t", index=False)

    computed = pd.DataFrame({"index": [1], "value": [3.0]})

    monkeypatch.setattr(
        "parcellate.interfaces.qsirecon.qsirecon.load_qsirecon_inputs",
        lambda *args, **kwargs: [recon],
    )
    monkeypatch.setattr(
        "parcellate.interfaces.qsirecon.qsirecon.plan_parcellation_workflow",
        lambda recon: {atlas: [scalar]},
    )
    monkeypatch.setattr(
        "parcellate.interfaces.qsirecon.qsirecon.run_parcellation_workflow",
        lambda *args, **kwargs: [
            ParcellationOutput(context, atlas, scalar, computed),
        ],
    )

    config = QSIReconConfig(input_root=tmp_path, output_dir=tmp_path, force=True)

    outputs = run_parcellations(config)

    assert len(outputs) == 1
    assert pd.read_csv(outputs[0], sep="\t").equals(computed)


def test_discover_scalar_maps_builds_definitions(tmp_path: Path) -> None:
    root = tmp_path
    scalar_path = (
        root / "derivatives" / "qsirecon-demo" / "sub-01" / "sub-01_desc-preproc_param-md_model-gqi_dwimap.nii.gz"
    )
    scalar_path.parent.mkdir(parents=True)
    scalar_path.touch()

    scalar_maps = discover_scalar_maps(root=tmp_path, subject="01", session=None)

    assert len(scalar_maps) == 1
    sm = scalar_maps[0]
    assert sm.param == "md"
    assert sm.model == "gqi"
    assert sm.space is None
    assert sm.recon_workflow == "demo"


def test_discover_atlases_falls_back_when_space_missing(tmp_path: Path) -> None:
    root = tmp_path
    (root / "derivatives").mkdir()
    (root / "derivatives" / "atlas-MyAtlas_dseg.nii.gz").touch()
    lut_file = FakeFile(
        root / "derivatives" / "atlas-MyAtlas_dseg.tsv",
        {"atlas": "MyAtlas", "extension": "tsv"},
    )
    (root / "derivatives" / "atlas-MyAtlas_dseg.tsv").touch()

    atlases = discover_atlases(root=root, space="SpaceA")

    assert len(atlases) == 1
    atlas = atlases[0]
    assert atlas.name == "MyAtlas"
    assert atlas.lut == Path(lut_file.path)
    assert atlas.space == "SpaceA"
    assert atlas.resolution is None


def test_load_qsirecon_inputs_builds_contexts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "derivatives" / "qsirecon-demo" / "sub-01" / "ses-S1").mkdir(parents=True)
    (tmp_path / "derivatives" / "qsirecon-demo" / "sub-01" / "ses-S1" / "sub-01_param-fa_dwimap.nii.gz").touch()
    (tmp_path / "atlas_dseg.nii.gz").touch()

    recon_inputs = load_qsirecon_inputs(root=tmp_path)

    assert len(recon_inputs) == 1
    recon = recon_inputs[0]
    assert recon.context.subject_id == "01"
    assert recon.context.session_id == "S1"
    assert recon.scalar_maps[0].param == "fa"
    assert recon.atlases[0].name == "atlas_dseg"


def test_space_match_is_case_insensitive() -> None:
    atlas = AtlasDefinition(name="atlas", nifti_path=Path("atlas.nii"), space="MNI")
    scalar = ScalarMapDefinition(name="map", nifti_path=Path("map.nii"), space="mni")
    assert _space_match(atlas, scalar)


def test_plan_qsirecon_parcellation_filters_by_space(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    atlas = AtlasDefinition(name="atlas", nifti_path=Path("atlas.nii"), space="MNI")
    matching = ScalarMapDefinition(name="map1", nifti_path=Path("map1.nii"), space="MNI")
    non_matching = ScalarMapDefinition(name="map2", nifti_path=Path("map2.nii"), space="other")

    recon = type(
        "Recon",
        (),
        {"atlases": [atlas], "scalar_maps": [matching, non_matching]},
    )()

    plan = plan_qsirecon_parcellation_workflow(recon)

    assert plan[atlas] == [matching]


def test_runner_creates_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[Path] = []

    class DummyParcellator:
        def __init__(
            self,
            atlas_img,
            lut=None,
            mask=None,
            background_label=0,
            resampling_target="data",
        ) -> None:
            calls.append(Path(atlas_img))

        def fit(self, scalar_img) -> None:
            calls.append(Path(scalar_img))

        def transform(self, scalar_img):
            calls.append(Path(scalar_img))
            return pd.DataFrame({"index": [1], "label": ["a"]})

    monkeypatch.setattr("parcellate.interfaces.runner.VolumetricParcellator", DummyParcellator)

    atlas = AtlasDefinition(name="atlas", nifti_path=tmp_path / "atlas.nii.gz")
    scalar1 = ScalarMapDefinition(name="map1", nifti_path=tmp_path / "map1.nii.gz")
    scalar2 = ScalarMapDefinition(name="map2", nifti_path=tmp_path / "map2.nii.gz")
    recon = type(
        "Recon",
        (),
        {
            "context": SubjectContext("01"),
            "atlases": [atlas],
            "scalar_maps": [scalar1, scalar2],
        },
    )()
    plan = {atlas: [scalar1, scalar2]}
    config = QSIReconConfig(input_root=tmp_path, output_dir=tmp_path)

    outputs = run_qsirecon_parcellation_workflow(recon=recon, plan=plan, config=config)

    assert len(outputs) == 2
    assert calls[0] == atlas.nifti_path
    assert outputs[0].scalar_map == scalar1
    assert outputs[1].scalar_map == scalar2


def test_runner_skips_atlas_on_space_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that runner skips atlas when scalar maps have mismatched spaces."""
    from parcellate.interfaces.runner import (
        ScalarMapSpaceMismatchError,
        _validate_scalar_map_spaces,
    )

    # Test the validation function directly
    scalar1 = ScalarMapDefinition(name="map1", nifti_path=tmp_path / "map1.nii.gz", space="MNI")
    scalar2 = ScalarMapDefinition(name="map2", nifti_path=tmp_path / "map2.nii.gz", space="native")

    with pytest.raises(ScalarMapSpaceMismatchError):
        _validate_scalar_map_spaces([scalar1, scalar2])


def test_runner_validates_matching_spaces(tmp_path: Path) -> None:
    """Test that validation passes when all scalar maps have the same space."""
    from parcellate.interfaces.runner import _validate_scalar_map_spaces

    scalar1 = ScalarMapDefinition(name="map1", nifti_path=tmp_path / "map1.nii.gz", space="MNI")
    scalar2 = ScalarMapDefinition(name="map2", nifti_path=tmp_path / "map2.nii.gz", space="MNI")

    # Should not raise
    _validate_scalar_map_spaces([scalar1, scalar2])


def test_runner_validates_empty_list(tmp_path: Path) -> None:
    """Test that validation passes for empty list."""
    from parcellate.interfaces.runner import _validate_scalar_map_spaces

    # Should not raise
    _validate_scalar_map_spaces([])


def test_scalar_map_space_mismatch_error_message() -> None:
    """Test ScalarMapSpaceMismatchError error message."""
    from parcellate.interfaces.runner import ScalarMapSpaceMismatchError

    error = ScalarMapSpaceMismatchError({"MNI", "native"})
    assert "MNI" in str(error)
    assert "native" in str(error)


def test_runner_continues_after_parcellator_init_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that runner continues processing other atlases when one fails to initialize."""

    class FailingParcellator:
        def __init__(self, atlas_img, **kwargs) -> None:
            if "failing" in str(atlas_img):
                msg = "Simulated init failure"
                raise ValueError(msg)

        def fit(self, scalar_img) -> None:
            pass

        def transform(self, scalar_img):
            return pd.DataFrame({"index": [1], "label": ["a"]})

    monkeypatch.setattr("parcellate.interfaces.runner.VolumetricParcellator", FailingParcellator)

    failing_atlas = AtlasDefinition(name="failing_atlas", nifti_path=tmp_path / "failing_atlas.nii.gz")
    working_atlas = AtlasDefinition(name="working_atlas", nifti_path=tmp_path / "working_atlas.nii.gz")
    scalar = ScalarMapDefinition(name="map", nifti_path=tmp_path / "map.nii.gz")
    recon = type("Recon", (), {"context": SubjectContext("01")})()
    plan = {failing_atlas: [scalar], working_atlas: [scalar]}
    config = QSIReconConfig(input_root=tmp_path, output_dir=tmp_path)

    outputs = run_qsirecon_parcellation_workflow(recon=recon, plan=plan, config=config)

    # Should have output from working atlas only
    assert len(outputs) == 1
    assert outputs[0].atlas == working_atlas


def test_runner_continues_after_transform_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that runner continues processing other scalar maps when one fails to transform."""
    transform_calls = []

    class PartiallyFailingParcellator:
        def __init__(self, atlas_img, **kwargs) -> None:
            pass

        def fit(self, scalar_img) -> None:
            pass

        def transform(self, scalar_img):
            transform_calls.append(str(scalar_img))
            if "failing" in str(scalar_img):
                msg = "Simulated transform failure"
                raise ValueError(msg)
            return pd.DataFrame({"index": [1], "label": ["a"]})

    monkeypatch.setattr(
        "parcellate.interfaces.runner.VolumetricParcellator",
        PartiallyFailingParcellator,
    )

    atlas = AtlasDefinition(name="atlas", nifti_path=tmp_path / "atlas.nii.gz")
    failing_scalar = ScalarMapDefinition(name="failing_map", nifti_path=tmp_path / "failing_map.nii.gz")
    working_scalar = ScalarMapDefinition(name="working_map", nifti_path=tmp_path / "working_map.nii.gz")
    recon = type("Recon", (), {"context": SubjectContext("01")})()
    plan = {atlas: [failing_scalar, working_scalar]}
    config = QSIReconConfig(input_root=tmp_path, output_dir=tmp_path)

    outputs = run_qsirecon_parcellation_workflow(recon=recon, plan=plan, config=config)

    # Should have output from working scalar only
    assert len(outputs) == 1
    assert outputs[0].scalar_map == working_scalar
    # Both transforms should have been attempted
    assert len(transform_calls) == 2


def test_runner_skips_empty_scalar_maps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that runner skips atlases with no scalar maps."""

    class DummyParcellator:
        def __init__(self, atlas_img, **kwargs) -> None:
            pass

        def fit(self, scalar_img) -> None:
            pass

        def transform(self, scalar_img):
            return pd.DataFrame({"index": [1], "label": ["a"]})

    monkeypatch.setattr("parcellate.interfaces.runner.VolumetricParcellator", DummyParcellator)

    atlas_with_maps = AtlasDefinition(name="atlas1", nifti_path=tmp_path / "atlas1.nii.gz")
    atlas_without_maps = AtlasDefinition(name="atlas2", nifti_path=tmp_path / "atlas2.nii.gz")
    scalar = ScalarMapDefinition(name="map", nifti_path=tmp_path / "map.nii.gz")
    recon = type("Recon", (), {"context": SubjectContext("01")})()
    plan = {atlas_with_maps: [scalar], atlas_without_maps: []}
    config = QSIReconConfig(input_root=tmp_path, output_dir=tmp_path)

    outputs = run_qsirecon_parcellation_workflow(recon=recon, plan=plan, config=config)

    assert len(outputs) == 1
    assert outputs[0].atlas == atlas_with_maps


def test_runner_skips_mismatched_space_atlas(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that runner skips atlas when scalar maps have different spaces."""

    class DummyParcellator:
        def __init__(self, atlas_img, **kwargs) -> None:
            pass

        def fit(self, scalar_img) -> None:
            pass

        def transform(self, scalar_img):
            return pd.DataFrame({"index": [1], "label": ["a"]})

    monkeypatch.setattr("parcellate.interfaces.runner.VolumetricParcellator", DummyParcellator)

    atlas = AtlasDefinition(name="atlas", nifti_path=tmp_path / "atlas.nii.gz")
    scalar_mni = ScalarMapDefinition(name="map1", nifti_path=tmp_path / "map1.nii.gz", space="MNI")
    scalar_native = ScalarMapDefinition(name="map2", nifti_path=tmp_path / "map2.nii.gz", space="native")
    recon = type("Recon", (), {"context": SubjectContext("01")})()
    plan = {atlas: [scalar_mni, scalar_native]}
    config = QSIReconConfig(input_root=tmp_path, output_dir=tmp_path)

    outputs = run_qsirecon_parcellation_workflow(recon=recon, plan=plan, config=config)

    # Should skip the atlas due to space mismatch
    assert len(outputs) == 0

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from parcellate.interfaces.qsirecon.loader import discover_atlases, discover_scalar_maps, load_qsirecon_inputs
from parcellate.interfaces.qsirecon.models import (
    AtlasDefinition,
    ParcellationOutput,
    QSIReconConfig,
    ScalarMapDefinition,
    SubjectContext,
)
from parcellate.interfaces.qsirecon.planner import _space_match, plan_qsirecon_parcellation_workflow
from parcellate.interfaces.qsirecon.qsirecon import (
    _as_list,
    _parse_log_level,
    _write_output,
    load_config,
    run_parcellations,
)
from parcellate.interfaces.qsirecon.runner import run_qsirecon_parcellation_workflow


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
            'log_level = "debug"',
        ])
    )

    config = load_config(cfg_path)

    assert config.input_root == Path("~/data").expanduser().resolve()
    assert config.output_dir == Path("outdir").expanduser().resolve()
    assert config.subjects == ["01", "02"]
    assert config.sessions == ["baseline"]
    assert config.mask == Path("mask.nii.gz").expanduser().resolve()
    assert config.log_level == logging.DEBUG


def test_write_output_creates_bids_like_path(tmp_path: Path) -> None:
    context = SubjectContext(subject_id="01", session_id="02")
    atlas = AtlasDefinition(name="atlasA", nifti_path=tmp_path / "atlas.nii.gz", resolution="2mm", space="MNI")
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

    out_path = _write_output(po, destination=tmp_path)

    expected_dir = tmp_path / "qsirecon-workflowX" / "sub-01" / "ses-02" / "dwi" / "atlas-atlasA"
    assert out_path.parent == expected_dir
    assert out_path.name.startswith("sub-01_ses-02_atlas-atlasA_space-MNI_res-2mm_model-gqi_param-odi_desc-smoothed")
    assert out_path.exists()
    written = pd.read_csv(out_path, sep="\t")
    assert written.equals(stats)


def test_run_parcellations_writes_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    context = SubjectContext("01")
    atlas = AtlasDefinition("atlas", nifti_path=tmp_path / "atlas.nii.gz")
    scalar = ScalarMapDefinition("map", nifti_path=tmp_path / "map.nii.gz", param="fa")
    stats = pd.DataFrame({"index": [1], "value": [1.0]})
    parcellation = ParcellationOutput(context=context, atlas=atlas, scalar_map=scalar, stats_table=stats)

    def fake_load_qsirecon_inputs(root: Path, subjects=None, sessions=None) -> list[Any]:
        return [
            type(
                "Recon",
                (),
                {"context": context, "atlases": [atlas], "scalar_maps": [scalar]},
            )()
        ]

    monkeypatch.setattr("parcellate.interfaces.qsirecon.qsirecon.load_qsirecon_inputs", fake_load_qsirecon_inputs)
    monkeypatch.setattr(
        "parcellate.interfaces.qsirecon.qsirecon.plan_qsirecon_parcellation_workflow",
        lambda recon: {atlas: [scalar]},
    )
    monkeypatch.setattr(
        "parcellate.interfaces.qsirecon.qsirecon.run_qsirecon_parcellation_workflow",
        lambda recon, plan, config: [parcellation],
    )

    config = QSIReconConfig(
        input_root=tmp_path,
        output_dir=tmp_path,
        subjects=["01"],
    )
    outputs = run_parcellations(config)

    assert len(outputs) == 1
    assert outputs[0].exists()


def test_discover_scalar_maps_builds_definitions(tmp_path: Path) -> None:
    root = tmp_path
    scalar_path = (
        root / "derivatives" / "qsirecon-demo" / "sub-01" / "sub-01_desc-preproc_param-md_model-gqi_dwimap.nii.gz"
    )
    scalar_path.parent.mkdir(parents=True)
    scalar_path.touch()
    files = [
        FakeFile(
            scalar_path,
            {
                "subject": "01",
                "suffix": "dwimap",
                "param": "md",
                "desc": "preproc",
                "model": "gqi",
                "space": "MNI",
            },
        )
    ]
    layout = FakeLayout(root=root, files=files, entities={"session": []}, subjects=["01"])

    scalar_maps = discover_scalar_maps(layout=layout, subject="01", session=None)

    assert len(scalar_maps) == 1
    sm = scalar_maps[0]
    assert sm.param == "md"
    assert sm.model == "gqi"
    assert sm.space == "MNI"
    assert sm.recon_workflow == "demo"


def test_discover_atlases_falls_back_when_space_missing(tmp_path: Path) -> None:
    root = tmp_path
    atlas_fallback = FakeFile(
        root / "derivatives" / "atlas-spaceB_dseg.nii.gz",
        {"suffix": "dseg", "space": "SpaceB", "segmentation": "MyAtlas", "res": "1mm", "extension": "nii.gz"},
    )
    lut_file = FakeFile(root / "derivatives" / "atlas-spaceB_dseg.tsv", {"atlas": "MyAtlas", "extension": "tsv"})

    class FallbackLayout(FakeLayout):
        def get(self, return_type: str = "object", **filters: Any) -> list[FakeFile]:
            if filters.get("space") == "SpaceA":
                return []
            return super().get(return_type=return_type, **filters)

    layout = FallbackLayout(
        root=root,
        files=[atlas_fallback, lut_file],
        entities={},
        subjects=["01"],
    )

    atlases = discover_atlases(layout=layout, space="SpaceA")

    assert len(atlases) == 1
    atlas = atlases[0]
    assert atlas.name == "MyAtlas"
    assert atlas.lut == Path(lut_file.path)
    assert atlas.space == "SpaceB"
    assert atlas.resolution == "1mm"


def test_load_qsirecon_inputs_builds_contexts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scalar_file = FakeFile(
        tmp_path / "derivatives" / "qsirecon-demo" / "sub-01" / "sub-01_param-fa_dwimap.nii.gz",
        {"subject": "01", "session": "S1", "suffix": "dwimap", "param": "fa", "space": "mni"},
    )
    atlas_file = FakeFile(
        tmp_path / "derivatives" / "atlas_dseg.nii.gz",
        {"suffix": "dseg", "space": "mni", "segmentation": "atlasA"},
    )

    def fake_layout(root: Path, validate: bool, derivatives: bool, config: list[str]) -> FakeLayout:
        return FakeLayout(
            root=root,
            files=[scalar_file, atlas_file],
            entities={"session": ["S1"]},
            subjects=["01"],
            sessions=["S1"],
        )

    monkeypatch.setattr("parcellate.interfaces.qsirecon.loader.BIDSLayout", fake_layout)

    recon_inputs = load_qsirecon_inputs(root=tmp_path)

    assert len(recon_inputs) == 1
    recon = recon_inputs[0]
    assert recon.context.subject_id == "01"
    assert recon.context.session_id == "S1"
    assert recon.scalar_maps[0].param == "fa"
    assert recon.atlases[0].name == "atlasA"


def test_space_match_is_case_insensitive() -> None:
    atlas = AtlasDefinition(name="atlas", nifti_path=Path("atlas.nii"), space="MNI")
    scalar = ScalarMapDefinition(name="map", nifti_path=Path("map.nii"), space="mni")
    assert _space_match(atlas, scalar)


def test_plan_qsirecon_parcellation_filters_by_space() -> None:
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
        def __init__(self, atlas_img, lut=None, mask=None, background_label=0, resampling_target="data") -> None:
            calls.append(Path(atlas_img))

        def fit(self, scalar_img) -> None:
            calls.append(Path(scalar_img))

        def transform(self, scalar_img):
            calls.append(Path(scalar_img))
            return pd.DataFrame({"index": [1], "label": ["a"]})

    monkeypatch.setattr("parcellate.interfaces.qsirecon.runner.VolumetricParcellator", DummyParcellator)

    atlas = AtlasDefinition(name="atlas", nifti_path=tmp_path / "atlas.nii.gz")
    scalar1 = ScalarMapDefinition(name="map1", nifti_path=tmp_path / "map1.nii.gz")
    scalar2 = ScalarMapDefinition(name="map2", nifti_path=tmp_path / "map2.nii.gz")
    recon = type(
        "Recon", (), {"context": SubjectContext("01"), "atlases": [atlas], "scalar_maps": [scalar1, scalar2]}
    )()
    plan = {atlas: [scalar1, scalar2]}
    config = QSIReconConfig(input_root=tmp_path, output_dir=tmp_path)

    outputs = run_qsirecon_parcellation_workflow(recon=recon, plan=plan, config=config)

    assert len(outputs) == 2
    assert calls[0] == atlas.nifti_path
    assert outputs[0].scalar_map == scalar1
    assert outputs[1].scalar_map == scalar2

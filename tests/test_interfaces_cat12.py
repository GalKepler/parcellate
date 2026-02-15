"""Tests for CAT12 interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from parcellate.interfaces.cat12.cat12 import (
    _build_output_path,
    _build_tiv_output_path,
    _extract_and_write_tiv,
    _find_first_tiv,
    _write_output,
    load_config,
    run_parcellations,
)
from parcellate.interfaces.cat12.cli import (
    SubjectSession,
    _parse_atlases_from_env,
    build_arg_parser,
    config_from_env,
    load_subjects_from_csv,
    main,
)
from parcellate.interfaces.cat12.loader import (
    TISSUE_PATTERNS,
    _build_search_path,
    _discover_sessions,
    _discover_subjects,
    _extract_desc,
    _scalar_name,
    discover_cat12_xml,
    discover_scalar_maps,
    extract_tiv_from_xml,
    load_cat12_inputs,
)
from parcellate.interfaces.cat12.models import (
    AtlasConfigurationError,
    AtlasDefinition,
    Cat12Config,
    ParcellationOutput,
    ScalarMapDefinition,
    SubjectContext,
    TissueType,
)
from parcellate.interfaces.planner import (
    _space_match,
)
from parcellate.interfaces.planner import (
    plan_parcellation_workflow as plan_cat12_parcellation_workflow,
)
from parcellate.interfaces.runner import (
    ScalarMapSpaceMismatchError,
    _validate_scalar_map_spaces,
)
from parcellate.interfaces.runner import (
    run_parcellation_workflow as run_cat12_parcellation_workflow,
)
from parcellate.interfaces.utils import _as_list, _parse_log_level, parse_atlases, write_parcellation_sidecar

# --- Model Tests ---


def test_tissue_type_enum_values() -> None:
    """Test TissueType enum has expected values."""
    assert TissueType.GM.value == "GM"
    assert TissueType.WM.value == "WM"
    assert TissueType.CT.value == "CT"


def test_tissue_patterns_defined() -> None:
    """Test tissue patterns are defined for all tissue types."""
    assert TissueType.GM in TISSUE_PATTERNS
    assert TissueType.WM in TISSUE_PATTERNS
    assert TissueType.CT in TISSUE_PATTERNS
    assert TISSUE_PATTERNS[TissueType.GM] == "mwp1*"
    assert TISSUE_PATTERNS[TissueType.WM] == "mwp2*"
    assert TISSUE_PATTERNS[TissueType.CT] == "wct*"


def test_subject_context_label() -> None:
    """Test SubjectContext label property."""
    context = SubjectContext(subject_id="01", session_id="02")
    assert context.label == "sub-01_ses-02"

    context_no_session = SubjectContext(subject_id="01")
    assert context_no_session.label == "sub-01"


def test_scalar_map_definition_with_tissue_type() -> None:
    """Test ScalarMapDefinition includes tissue_type field."""
    scalar = ScalarMapDefinition(
        name="GM-mwp1sub01",
        nifti_path=Path("/path/to/mwp1sub01.nii.gz"),
        tissue_type=TissueType.GM,
        space="MNI152NLin2009cAsym",
    )
    assert scalar.tissue_type == TissueType.GM


def test_cat12_config_has_atlases_field() -> None:
    """Test Cat12Config includes atlases field."""
    config = Cat12Config(
        input_root=Path("/input"),
        output_dir=Path("/output"),
        atlases=[],
    )
    assert config.atlases == []


# --- Loader Tests ---


def test_discover_subjects(tmp_path: Path) -> None:
    """Test subject discovery from directory structure."""
    (tmp_path / "sub-01").mkdir()
    (tmp_path / "sub-02").mkdir()
    (tmp_path / "not_a_subject").mkdir()

    subjects = _discover_subjects(tmp_path)

    assert subjects == ["01", "02"]


def test_discover_sessions(tmp_path: Path) -> None:
    """Test session discovery from directory structure."""
    subject_dir = tmp_path / "sub-01"
    subject_dir.mkdir()
    (subject_dir / "ses-01").mkdir()
    (subject_dir / "ses-02").mkdir()
    (subject_dir / "anat").mkdir()

    sessions = _discover_sessions(tmp_path, "01")

    assert sessions == ["01", "02"]


def test_discover_sessions_returns_none_when_no_sessions(tmp_path: Path) -> None:
    """Test session discovery returns [None] when no sessions exist."""
    subject_dir = tmp_path / "sub-01"
    subject_dir.mkdir()

    sessions = _discover_sessions(tmp_path, "01")

    assert sessions == [None]


def test_build_search_path_with_session(tmp_path: Path) -> None:
    """Test search path building with session."""
    path = _build_search_path(tmp_path, "01", "02")
    assert path == tmp_path / "sub-01" / "ses-02" / "anat"


def test_build_search_path_without_session(tmp_path: Path) -> None:
    """Test search path building without session."""
    path = _build_search_path(tmp_path, "01", None)
    assert path == tmp_path / "sub-01" / "anat"


def test_discover_scalar_maps_finds_gm_files(tmp_path: Path) -> None:
    """Test scalar map discovery for GM tissue files."""
    anat_dir = tmp_path / "sub-01" / "anat"
    anat_dir.mkdir(parents=True)
    gm_file = anat_dir / "mwp1sub-01_T1w.nii.gz"
    gm_file.touch()

    scalar_maps = discover_scalar_maps(tmp_path, "01", None)

    assert len(scalar_maps) == 1
    assert scalar_maps[0].tissue_type == TissueType.GM
    assert scalar_maps[0].nifti_path == gm_file


def test_discover_scalar_maps_finds_wm_files(tmp_path: Path) -> None:
    """Test scalar map discovery for WM tissue files."""
    anat_dir = tmp_path / "sub-01" / "anat"
    anat_dir.mkdir(parents=True)
    wm_file = anat_dir / "mwp2sub-01_T1w.nii.gz"
    wm_file.touch()

    scalar_maps = discover_scalar_maps(tmp_path, "01", None)

    assert len(scalar_maps) == 1
    assert scalar_maps[0].tissue_type == TissueType.WM


def test_discover_scalar_maps_finds_ct_files(tmp_path: Path) -> None:
    """Test scalar map discovery for CT tissue files."""
    anat_dir = tmp_path / "sub-01" / "anat"
    anat_dir.mkdir(parents=True)
    ct_file = anat_dir / "wctT1w.nii.gz"
    ct_file.touch()

    scalar_maps = discover_scalar_maps(tmp_path, "01", None)

    assert len(scalar_maps) == 1
    assert scalar_maps[0].tissue_type == TissueType.CT


def test_discover_scalar_maps_finds_all_tissue_types(tmp_path: Path) -> None:
    """Test scalar map discovery finds all tissue types."""
    anat_dir = tmp_path / "sub-01" / "anat"
    anat_dir.mkdir(parents=True)
    (anat_dir / "mwp1sub-01_T1w.nii.gz").touch()
    (anat_dir / "mwp2sub-01_T1w.nii.gz").touch()
    (anat_dir / "wctT1w.nii.gz").touch()

    scalar_maps = discover_scalar_maps(tmp_path, "01", None)

    assert len(scalar_maps) == 3
    tissue_types = {sm.tissue_type for sm in scalar_maps}
    assert tissue_types == {TissueType.GM, TissueType.WM, TissueType.CT}


def test_discover_scalar_maps_with_session(tmp_path: Path) -> None:
    """Test scalar map discovery with session."""
    anat_dir = tmp_path / "sub-01" / "ses-02" / "anat"
    anat_dir.mkdir(parents=True)
    (anat_dir / "mwp1sub-01_ses-02_T1w.nii.gz").touch()

    scalar_maps = discover_scalar_maps(tmp_path, "01", "02")

    assert len(scalar_maps) == 1


def test_discover_scalar_maps_returns_empty_when_path_missing(tmp_path: Path) -> None:
    """Test scalar map discovery returns empty list when path doesn't exist."""
    scalar_maps = discover_scalar_maps(tmp_path, "nonexistent", None)
    assert scalar_maps == []


def test_scalar_name_construction(tmp_path: Path) -> None:
    """Test scalar name construction."""
    nii_path = tmp_path / "mwp1sub-01_T1w.nii.gz"
    name = _scalar_name(nii_path, TissueType.GM)
    assert name == "GM-mwp1sub-01_T1w"


def test_extract_desc(tmp_path: Path) -> None:
    """Test description extraction."""
    nii_path = tmp_path / "mwp1sub-01_T1w.nii.gz"
    desc = _extract_desc(nii_path)
    assert desc == "mwp1sub-01_T1w"


def test_load_cat12_inputs_builds_recon_inputs(tmp_path: Path) -> None:
    """Test load_cat12_inputs builds correct recon inputs."""
    anat_dir = tmp_path / "sub-01" / "anat"
    anat_dir.mkdir(parents=True)
    (anat_dir / "mwp1sub-01_T1w.nii.gz").touch()

    atlas = AtlasDefinition(
        name="TestAtlas",
        nifti_path=tmp_path / "atlas.nii.gz",
        space="MNI152NLin2009cAsym",
    )

    recon_inputs = load_cat12_inputs(tmp_path, [atlas])

    assert len(recon_inputs) == 1
    assert recon_inputs[0].context.subject_id == "01"
    assert len(recon_inputs[0].scalar_maps) == 1
    assert recon_inputs[0].atlases == [atlas]


def test_load_cat12_inputs_filters_subjects(tmp_path: Path) -> None:
    """Test load_cat12_inputs filters by specified subjects."""
    for subj in ["01", "02", "03"]:
        anat_dir = tmp_path / f"sub-{subj}" / "anat"
        anat_dir.mkdir(parents=True)
        (anat_dir / f"mwp1sub-{subj}_T1w.nii.gz").touch()

    atlas = AtlasDefinition(name="Atlas", nifti_path=tmp_path / "atlas.nii.gz", space="MNI152NLin2009cAsym")
    recon_inputs = load_cat12_inputs(tmp_path, [atlas], subjects=["01", "03"])

    assert len(recon_inputs) == 2
    subject_ids = {ri.context.subject_id for ri in recon_inputs}
    assert subject_ids == {"01", "03"}


def test_load_cat12_inputs_returns_empty_when_no_subjects(tmp_path: Path) -> None:
    """Test load_cat12_inputs returns empty when no subjects found."""
    atlas = AtlasDefinition(name="Atlas", nifti_path=tmp_path / "atlas.nii.gz", space="MNI152NLin2009cAsym")
    recon_inputs = load_cat12_inputs(tmp_path, [atlas])
    assert recon_inputs == []


# --- Planner Tests ---


def test_space_match_is_case_insensitive() -> None:
    """Test space matching is case insensitive."""
    atlas = AtlasDefinition(name="atlas", nifti_path=Path("atlas.nii"), space="MNI152NLin2009cAsym")
    scalar = ScalarMapDefinition(name="map", nifti_path=Path("map.nii"), space="mni152nlin2009casym")
    assert _space_match(atlas, scalar)


def test_space_match_returns_false_when_no_space() -> None:
    """Test space match returns False when space is missing."""
    atlas = AtlasDefinition(name="atlas", nifti_path=Path("atlas.nii"), space=None)
    scalar = ScalarMapDefinition(name="map", nifti_path=Path("map.nii"), space="MNI")
    assert not _space_match(atlas, scalar)


def test_plan_cat12_parcellation_filters_by_space() -> None:
    """Test plan_cat12_parcellation_workflow filters scalar maps by space."""
    atlas = AtlasDefinition(name="atlas", nifti_path=Path("atlas.nii"), space="MNI152NLin2009cAsym")
    matching = ScalarMapDefinition(name="map1", nifti_path=Path("map1.nii"), space="MNI152NLin2009cAsym")
    non_matching = ScalarMapDefinition(name="map2", nifti_path=Path("map2.nii"), space="other")

    recon = type(
        "Recon",
        (),
        {"atlases": [atlas], "scalar_maps": [matching, non_matching]},
    )()

    plan = plan_cat12_parcellation_workflow(recon)

    assert plan[atlas] == [matching]


# --- Runner Tests ---


def test_validate_scalar_map_spaces_passes_for_matching(tmp_path: Path) -> None:
    """Test validation passes when all scalar maps have same space."""
    scalar1 = ScalarMapDefinition(name="map1", nifti_path=tmp_path / "map1.nii.gz", space="MNI")
    scalar2 = ScalarMapDefinition(name="map2", nifti_path=tmp_path / "map2.nii.gz", space="MNI")
    _validate_scalar_map_spaces([scalar1, scalar2])  # Should not raise


def test_validate_scalar_map_spaces_raises_for_mismatch(tmp_path: Path) -> None:
    """Test validation raises when scalar maps have different spaces."""
    scalar1 = ScalarMapDefinition(name="map1", nifti_path=tmp_path / "map1.nii.gz", space="MNI")
    scalar2 = ScalarMapDefinition(name="map2", nifti_path=tmp_path / "map2.nii.gz", space="native")

    with pytest.raises(ScalarMapSpaceMismatchError):
        _validate_scalar_map_spaces([scalar1, scalar2])


def test_validate_scalar_map_spaces_passes_for_empty() -> None:
    """Test validation passes for empty list."""
    _validate_scalar_map_spaces([])  # Should not raise


def test_scalar_map_space_mismatch_error_message() -> None:
    """Test error message contains space names."""
    error = ScalarMapSpaceMismatchError({"MNI", "native"})
    assert "MNI" in str(error)
    assert "native" in str(error)


class DummyParcellator:
    def __init__(
        self,
        atlas_img,
        lut=None,
        mask=None,
        background_label=0,
        resampling_target="data",
    ) -> None:
        pass

    def fit(self, scalar_img) -> None:
        pass

    def transform(self, scalar_img):
        return pd.DataFrame({"index": [1], "label": ["a"]})


def test_runner_creates_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test runner creates parcellation outputs."""
    monkeypatch.setattr("parcellate.interfaces.runner.VolumetricParcellator", DummyParcellator)

    atlas = AtlasDefinition(name="atlas", nifti_path=tmp_path / "atlas.nii.gz")
    scalar1 = ScalarMapDefinition(name="map1", nifti_path=tmp_path / "map1.nii.gz", tissue_type=TissueType.GM)
    scalar2 = ScalarMapDefinition(name="map2", nifti_path=tmp_path / "map2.nii.gz", tissue_type=TissueType.WM)
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
    config = Cat12Config(input_root=tmp_path, output_dir=tmp_path)

    outputs = run_cat12_parcellation_workflow(recon=recon, plan=plan, config=config)

    assert len(outputs) == 2


def test_runner_skips_empty_scalar_maps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test runner skips atlases with no scalar maps."""

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
    config = Cat12Config(input_root=tmp_path, output_dir=tmp_path)

    outputs = run_cat12_parcellation_workflow(recon=recon, plan=plan, config=config)

    assert len(outputs) == 1
    assert outputs[0].atlas == atlas_with_maps


# --- Config Tests ---


def test_parse_log_level_handles_common_inputs() -> None:
    """Test log level parsing."""
    assert _parse_log_level(None) == logging.INFO
    assert _parse_log_level(logging.DEBUG) == logging.DEBUG
    assert _parse_log_level("debug") == logging.DEBUG
    assert _parse_log_level("INFO") == logging.INFO


def test_as_list_normalizes_inputs() -> None:
    """Test list normalization."""
    assert _as_list(None) is None
    assert _as_list("one") == ["one"]
    assert _as_list(["one", "two"]) == ["one", "two"]


def test_parse_atlases() -> None:
    """Test atlas configuration parsing."""
    atlas_configs = [
        {
            "name": "Schaefer400",
            "path": "/path/to/schaefer400.nii.gz",
            "lut": "/path/to/schaefer400.tsv",
            "space": "MNI152NLin2009cAsym",
        },
        {
            "name": "AAL",
            "path": "/path/to/aal.nii.gz",
        },
    ]

    atlases = parse_atlases(atlas_configs, default_space="MNI152NLin2009cAsym")

    assert len(atlases) == 2
    assert atlases[0].name == "Schaefer400"
    assert atlases[0].lut is not None
    assert atlases[1].name == "AAL"
    assert atlases[1].space == "MNI152NLin2009cAsym"  # Should get default space
    assert atlases[1].space == "MNI152NLin2009cAsym"  # default


def test_parse_atlases_skips_invalid() -> None:
    """Test atlas parsing skips entries with missing name or path."""
    atlas_configs = [
        {"name": "Valid", "path": "/path/to/valid.nii.gz"},
        {"name": "MissingPath"},
        {"path": "/path/to/missing_name.nii.gz"},
    ]

    atlases = parse_atlases(atlas_configs)

    assert len(atlases) == 1
    assert atlases[0].name == "Valid"


def test_load_config_reads_toml(tmp_path: Path) -> None:
    """Test configuration loading from TOML file."""
    cfg_path = tmp_path / "cat12.toml"
    cfg_path.write_text(
        "\n".join([
            'input_root = "~/data"',
            'output_dir = "outdir"',
            'subjects = ["01", "02"]',
            'sessions = ["baseline"]',
            'mask = "mask.nii.gz"',
            "force = true",
            'log_level = "debug"',
            "",
            "[[atlases]]",
            'name = "Schaefer400"',
            'path = "/path/to/schaefer400.nii.gz"',
            'space = "MNI152NLin2009cAsym"',
        ])
    )

    # Create argparse.Namespace with config path
    import argparse

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
    assert len(config.atlases) == 1
    assert config.atlases[0].name == "Schaefer400"


# --- Output Path Tests ---


def test_build_output_path_with_tissue_type(tmp_path: Path) -> None:
    """Test output path building with tissue type."""
    context = SubjectContext(subject_id="01", session_id="02")
    atlas = AtlasDefinition(name="atlasA", nifti_path=tmp_path / "atlas.nii.gz", space="MNI152NLin2009cAsym")
    scalar = ScalarMapDefinition(
        name="GM-mwp1sub01",
        nifti_path=tmp_path / "mwp1sub01.nii.gz",
        tissue_type=TissueType.GM,
        space="MNI152NLin2009cAsym",
    )

    out_path = _build_output_path(context, atlas, scalar, tmp_path)

    expected_dir = tmp_path / "cat12" / "sub-01" / "ses-02" / "anat" / "atlas-atlasA"
    assert out_path.parent == expected_dir
    assert "tissue-GM" in out_path.name
    assert out_path.name.endswith("_parc.tsv")


def test_build_output_path_without_session(tmp_path: Path) -> None:
    """Test output path building without session."""
    context = SubjectContext(subject_id="01")
    atlas = AtlasDefinition(name="atlasA", nifti_path=tmp_path / "atlas.nii.gz", space="MNI")
    scalar = ScalarMapDefinition(name="map", nifti_path=tmp_path / "map.nii.gz", tissue_type=TissueType.WM)

    out_path = _build_output_path(context, atlas, scalar, tmp_path)

    expected_dir = tmp_path / "cat12" / "sub-01" / "anat" / "atlas-atlasA"
    assert out_path.parent == expected_dir


def test_write_output_creates_file(tmp_path: Path) -> None:
    """Test _write_output creates the output TSV and JSON sidecar."""
    context = SubjectContext(subject_id="01")
    atlas = AtlasDefinition(name="atlasA", nifti_path=tmp_path / "atlas.nii.gz", space="MNI")
    scalar = ScalarMapDefinition(
        name="GM-mwp1sub01",
        nifti_path=tmp_path / "mwp1sub01.nii.gz",
        tissue_type=TissueType.GM,
    )
    stats = pd.DataFrame({"index": [1], "value": [3.14]})
    po = ParcellationOutput(context=context, atlas=atlas, scalar_map=scalar, stats_table=stats)
    config = Cat12Config(input_root=tmp_path, output_dir=tmp_path)

    out_path = _write_output(po, destination=tmp_path, config=config)

    assert out_path.exists()
    written = pd.read_csv(out_path, sep="\t")
    assert written.equals(stats)

    # JSON sidecar should also exist
    json_path = out_path.with_suffix(".json")
    assert json_path.exists()


# --- Sidecar Tests ---


def test_write_parcellation_sidecar_creates_json(tmp_path: Path) -> None:
    """Test write_parcellation_sidecar creates a valid JSON file."""
    tsv_path = tmp_path / "sub-01_atlas-test_parc.tsv"
    tsv_path.touch()

    json_path = write_parcellation_sidecar(
        tsv_path=tsv_path,
        original_file=tmp_path / "mwp1sub01.nii.gz",
        atlas_name="test_atlas",
        atlas_image=tmp_path / "atlas.nii.gz",
        atlas_lut=tmp_path / "atlas.tsv",
        mask=None,
        space="MNI152NLin2009cAsym",
        resampling_target="data",
        background_label=0,
    )

    assert json_path.exists()
    assert json_path.suffix == ".json"
    assert json_path.stem == tsv_path.stem


def test_write_parcellation_sidecar_content(tmp_path: Path) -> None:
    """Test sidecar JSON contains all expected fields."""
    tsv_path = tmp_path / "sub-01_atlas-test_parc.tsv"
    tsv_path.touch()
    original = tmp_path / "mwp1sub01.nii.gz"
    atlas_img = tmp_path / "atlas.nii.gz"
    atlas_lut = tmp_path / "atlas.tsv"

    write_parcellation_sidecar(
        tsv_path=tsv_path,
        original_file=original,
        atlas_name="my_atlas",
        atlas_image=atlas_img,
        atlas_lut=atlas_lut,
        mask=Path("/path/to/mask.nii.gz"),
        space="MNI152NLin2009cAsym",
        resampling_target="data",
        background_label=0,
    )

    data = json.loads(tsv_path.with_suffix(".json").read_text())
    assert data["original_file"] == str(original)
    assert data["mask"] == "/path/to/mask.nii.gz"
    assert data["parcellation_scheme"]["name"] == "my_atlas"
    assert data["parcellation_scheme"]["image"] == str(atlas_img)
    assert data["parcellation_scheme"]["lut"] == str(atlas_lut)
    assert data["space"] == "MNI152NLin2009cAsym"
    assert data["resampling_target"] == "data"
    assert data["background_label"] == 0
    assert "software_version" in data
    assert "timestamp" in data


def test_write_parcellation_sidecar_null_mask(tmp_path: Path) -> None:
    """Test sidecar JSON handles None mask correctly."""
    tsv_path = tmp_path / "sub-01_parc.tsv"
    tsv_path.touch()

    write_parcellation_sidecar(
        tsv_path=tsv_path,
        original_file=tmp_path / "scalar.nii.gz",
        atlas_name="atlas",
        atlas_image=tmp_path / "atlas.nii.gz",
        atlas_lut=None,
        mask=None,
    )

    data = json.loads(tsv_path.with_suffix(".json").read_text())
    assert data["mask"] is None
    assert data["parcellation_scheme"]["lut"] is None


def test_write_parcellation_sidecar_builtin_mask_string(tmp_path: Path) -> None:
    """Test sidecar JSON preserves builtin mask name string."""
    tsv_path = tmp_path / "sub-01_parc.tsv"
    tsv_path.touch()

    write_parcellation_sidecar(
        tsv_path=tsv_path,
        original_file=tmp_path / "scalar.nii.gz",
        atlas_name="atlas",
        atlas_image=tmp_path / "atlas.nii.gz",
        atlas_lut=None,
        mask="gm",
    )

    data = json.loads(tsv_path.with_suffix(".json").read_text())
    assert data["mask"] == "gm"


# --- Integration Tests ---


def test_run_parcellations_writes_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test run_parcellations writes outputs correctly."""
    context = SubjectContext("01")
    atlas = AtlasDefinition(name="atlas", nifti_path=tmp_path / "atlas.nii.gz", space="MNI152NLin2009cAsym")
    scalar = ScalarMapDefinition(
        name="GM-mwp1sub01",
        nifti_path=tmp_path / "mwp1sub01.nii.gz",
        tissue_type=TissueType.GM,
        space="MNI152NLin2009cAsym",
    )
    recon = type("Recon", (), {"context": context, "atlases": [atlas], "scalar_maps": [scalar]})()  # type: ignore[call]

    monkeypatch.setattr(
        "parcellate.interfaces.cat12.cat12.load_cat12_inputs",
        lambda *args, **kwargs: [recon],
    )
    monkeypatch.setattr(
        "parcellate.interfaces.cat12.cat12.plan_parcellation_workflow",
        lambda recon: {atlas: [scalar]},
    )
    monkeypatch.setattr(
        "parcellate.interfaces.cat12.cat12.run_parcellation_workflow",
        lambda *args, **kwargs: [
            ParcellationOutput(context, atlas, scalar, pd.DataFrame({"index": [1], "value": [2.0]}))
        ],
    )

    config = Cat12Config(input_root=tmp_path, output_dir=tmp_path, atlases=[atlas], force=True)
    outputs = run_parcellations(config)
    assert len(outputs) == 1
    out_path = outputs[0]
    assert out_path.exists()
    written = pd.read_csv(out_path, sep="\t")
    assert written.equals(pd.DataFrame({"index": [1], "value": [2.0]}))


def test_run_parcellations_returns_empty_when_no_atlases(tmp_path: Path) -> None:
    """Test run_parcellations raises error when no atlases configured."""
    import pytest

    config = Cat12Config(
        input_root=tmp_path,
        output_dir=tmp_path,
        atlases=None,
    )
    with pytest.raises(AtlasConfigurationError):
        run_parcellations(config)


def test_run_parcellations_reuses_existing_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test run_parcellations reuses existing outputs when force=False."""
    context = SubjectContext("01")
    atlas = AtlasDefinition(name="atlas", nifti_path=tmp_path / "atlas.nii.gz", space="MNI152NLin2009cAsym")
    scalar = ScalarMapDefinition(
        name="map",
        nifti_path=tmp_path / "map.nii.gz",
        tissue_type=TissueType.GM,
        space="MNI152NLin2009cAsym",
    )
    recon = type("Recon", (), {"context": context, "atlases": [atlas], "scalar_maps": [scalar]})()

    out_path = _build_output_path(context, atlas, scalar, tmp_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame({"index": [1], "value": [2.0]})
    existing.to_csv(out_path, sep="\t", index=False)

    monkeypatch.setattr(
        "parcellate.interfaces.cat12.cat12.load_cat12_inputs",
        lambda *args, **kwargs: [recon],
    )
    monkeypatch.setattr(
        "parcellate.interfaces.cat12.cat12.plan_parcellation_workflow",
        lambda recon: {atlas: [scalar]},
    )
    monkeypatch.setattr(
        "parcellate.interfaces.cat12.cat12.run_parcellation_workflow",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Should not run when outputs exist")),
    )

    config = Cat12Config(input_root=tmp_path, output_dir=tmp_path, atlases=[atlas], force=False)

    outputs = run_parcellations(config)

    assert outputs == [out_path]
    assert pd.read_csv(out_path, sep="\t").equals(existing)


# --- TIV Tests ---


_SIMPLE_TIV_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<cat_report>
  <subjectmeasures>
    <vol_TIV>1234.567</vol_TIV>
  </subjectmeasures>
</cat_report>
"""

_DESCRIPTION_THEN_NUMERIC_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<cat_report>
  <subjectmeasures>
    <vol_TIV>Total Intracranial Volume</vol_TIV>
  </subjectmeasures>
  <subjectmeasures>
    <vol_TIV>9876.0</vol_TIV>
  </subjectmeasures>
</cat_report>
"""


def test_discover_cat12_xml_finds_files(tmp_path: Path) -> None:
    """Test XML discovery finds cat_*.xml files in the anat directory."""
    anat_dir = tmp_path / "sub-01" / "anat"
    anat_dir.mkdir(parents=True)
    xml_file = anat_dir / "cat_mwp1sub-01_T1w.xml"
    xml_file.touch()
    other = anat_dir / "other.xml"
    other.touch()

    found = discover_cat12_xml(tmp_path, "01", None)

    assert found == [xml_file]


def test_discover_cat12_xml_with_session(tmp_path: Path) -> None:
    """Test XML discovery respects session subdirectory."""
    anat_dir = tmp_path / "sub-01" / "ses-02" / "anat"
    anat_dir.mkdir(parents=True)
    xml_file = anat_dir / "cat_mwp1sub-01_ses-02_T1w.xml"
    xml_file.touch()

    found = discover_cat12_xml(tmp_path, "01", "02")

    assert found == [xml_file]


def test_discover_cat12_xml_returns_empty_when_no_files(tmp_path: Path) -> None:
    """Test XML discovery returns empty list when no XML files exist."""
    anat_dir = tmp_path / "sub-01" / "anat"
    anat_dir.mkdir(parents=True)

    found = discover_cat12_xml(tmp_path, "01", None)

    assert found == []


def test_discover_cat12_xml_returns_empty_when_path_missing(tmp_path: Path) -> None:
    """Test XML discovery returns empty list when the anat directory is missing."""
    found = discover_cat12_xml(tmp_path, "nonexistent", None)
    assert found == []


def test_extract_tiv_from_xml_valid(tmp_path: Path) -> None:
    """Test TIV extraction returns the numeric value from a well-formed XML."""
    xml_path = tmp_path / "cat_sub-01.xml"
    xml_path.write_text(_SIMPLE_TIV_XML)

    tiv = extract_tiv_from_xml(xml_path)

    assert tiv == pytest.approx(1234.567)


def test_extract_tiv_from_xml_description_text_skipped(tmp_path: Path) -> None:
    """Test TIV extraction skips non-numeric vol_TIV and finds the numeric one."""
    xml_path = tmp_path / "cat_sub-01.xml"
    xml_path.write_text(_DESCRIPTION_THEN_NUMERIC_XML)

    tiv = extract_tiv_from_xml(xml_path)

    assert tiv == pytest.approx(9876.0)


def test_extract_tiv_from_xml_malformed(tmp_path: Path) -> None:
    """Test TIV extraction returns None for malformed XML."""
    xml_path = tmp_path / "cat_sub-01.xml"
    xml_path.write_text("<unclosed_tag>")

    tiv = extract_tiv_from_xml(xml_path)

    assert tiv is None


def test_extract_tiv_from_xml_no_tiv_element(tmp_path: Path) -> None:
    """Test TIV extraction returns None when vol_TIV element is absent."""
    xml_path = tmp_path / "cat_sub-01.xml"
    xml_path.write_text("<cat_report><subjectmeasures><other>1.0</other></subjectmeasures></cat_report>")

    tiv = extract_tiv_from_xml(xml_path)

    assert tiv is None


def test_build_tiv_output_path_with_session(tmp_path: Path) -> None:
    """Test TIV output path includes session subdirectory."""
    context = SubjectContext(subject_id="01", session_id="02")
    path = _build_tiv_output_path(context, tmp_path)

    expected = tmp_path / "cat12" / "sub-01" / "ses-02" / "anat" / "sub-01_ses-02_tiv.tsv"
    assert path == expected


def test_build_tiv_output_path_without_session(tmp_path: Path) -> None:
    """Test TIV output path without session."""
    context = SubjectContext(subject_id="01")
    path = _build_tiv_output_path(context, tmp_path)

    expected = tmp_path / "cat12" / "sub-01" / "anat" / "sub-01_tiv.tsv"
    assert path == expected


def test_extract_and_write_tiv_creates_file(tmp_path: Path) -> None:
    """Test _extract_and_write_tiv creates a TSV with vol_TIV values."""
    anat_dir = tmp_path / "sub-01" / "anat"
    anat_dir.mkdir(parents=True)
    xml_path = anat_dir / "cat_sub-01.xml"
    xml_path.write_text(_SIMPLE_TIV_XML)

    context = SubjectContext(subject_id="01")
    out_path = _extract_and_write_tiv(root=tmp_path, context=context, destination=tmp_path)

    assert out_path is not None
    assert out_path.exists()
    data = pd.read_csv(out_path, sep="\t")
    assert list(data.columns) == ["source_file", "vol_TIV"]
    assert data["vol_TIV"].iloc[0] == pytest.approx(1234.567)
    assert data["source_file"].iloc[0] == str(xml_path)


def test_extract_and_write_tiv_returns_none_when_no_xml(tmp_path: Path) -> None:
    """Test _extract_and_write_tiv returns None when no XML files are found."""
    context = SubjectContext(subject_id="01")
    (tmp_path / "sub-01" / "anat").mkdir(parents=True)

    out_path = _extract_and_write_tiv(root=tmp_path, context=context, destination=tmp_path)

    assert out_path is None


def test_find_first_tiv_returns_first_valid(tmp_path: Path) -> None:
    """Test _find_first_tiv returns the first non-None TIV value."""
    anat_dir = tmp_path / "sub-01" / "anat"
    anat_dir.mkdir(parents=True)
    (anat_dir / "cat_a.xml").write_text(_SIMPLE_TIV_XML)

    tiv = _find_first_tiv(tmp_path, "01", None)

    assert tiv == pytest.approx(1234.567)


def test_find_first_tiv_returns_none_when_no_xml(tmp_path: Path) -> None:
    """Test _find_first_tiv returns None when no XML files are present."""
    (tmp_path / "sub-01" / "anat").mkdir(parents=True)
    assert _find_first_tiv(tmp_path, "01", None) is None


# --- CLI Tests ---


def test_load_subjects_from_csv(tmp_path: Path) -> None:
    """Test loading subjects from CSV file."""
    csv_path = tmp_path / "subjects.csv"
    csv_path.write_text("subject_code,session_id\n01,baseline\n02,followup\n03,\n")

    subjects = load_subjects_from_csv(csv_path)

    assert len(subjects) == 3
    assert subjects[0].subject_code == "0001"
    assert subjects[0].session_id == "baseline"
    assert subjects[1].subject_code == "0002"
    assert subjects[1].session_id == "followup"
    assert subjects[2].subject_code == "0003"
    assert subjects[2].session_id is None


def test_load_subjects_from_csv_without_session(tmp_path: Path) -> None:
    """Test loading subjects from CSV without session column."""
    csv_path = tmp_path / "subjects.csv"
    csv_path.write_text("subject_code\n01\n02\n")

    subjects = load_subjects_from_csv(csv_path)

    assert len(subjects) == 2
    assert subjects[0].subject_code == "0001"
    assert subjects[0].session_id is None


def test_load_subjects_from_csv_missing_column(tmp_path: Path) -> None:
    """Test error when CSV is missing required column."""
    csv_path = tmp_path / "subjects.csv"
    csv_path.write_text("wrong_column,session_id\n01,baseline\n")

    with pytest.raises(ValueError, match="subject_code"):
        load_subjects_from_csv(csv_path)


def test_cli_parse_log_level() -> None:
    """Test log level parsing in CLI (uses shared _parse_log_level)."""
    assert _parse_log_level(None) == logging.INFO
    assert _parse_log_level("DEBUG") == logging.DEBUG
    assert _parse_log_level("warning") == logging.WARNING


def test_parse_atlases_from_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test atlas parsing with no environment variables."""
    monkeypatch.delenv("CAT12_ATLAS_PATHS", raising=False)
    atlases = _parse_atlases_from_env()
    assert atlases == []


def test_parse_atlases_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test atlas parsing from environment variables."""
    atlas1 = tmp_path / "atlas1.nii.gz"
    atlas2 = tmp_path / "atlas2.nii.gz"
    atlas1.touch()
    atlas2.touch()

    monkeypatch.setenv("CAT12_ATLAS_PATHS", f"{atlas1},{atlas2}")
    monkeypatch.setenv("CAT12_ATLAS_NAMES", "Schaefer400,AAL")
    monkeypatch.setenv("CAT12_ATLAS_SPACE", "MNI152NLin2009cAsym")

    atlases = _parse_atlases_from_env()

    assert len(atlases) == 2
    assert atlases[0].name == "Schaefer400"
    assert atlases[1].name == "AAL"
    assert atlases[0].space == "MNI152NLin2009cAsym"


def test_config_from_env_missing_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error when CAT12_ROOT is not set."""
    monkeypatch.delenv("CAT12_ROOT", raising=False)

    with pytest.raises(ValueError, match="CAT12_ROOT"):
        config_from_env()


def test_config_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test config creation from environment variables."""
    atlas_path = tmp_path / "atlas.nii.gz"
    atlas_path.touch()

    monkeypatch.setenv("CAT12_ROOT", str(tmp_path))
    monkeypatch.setenv("CAT12_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("CAT12_ATLAS_PATHS", str(atlas_path))
    monkeypatch.setenv("CAT12_ATLAS_NAMES", "TestAtlas")
    monkeypatch.setenv("CAT12_LOG_LEVEL", "DEBUG")

    config = config_from_env()

    assert config.input_root == tmp_path
    assert config.output_dir == tmp_path / "output"
    assert len(config.atlases) == 1
    assert config.atlases[0].name == "TestAtlas"
    assert config.log_level == logging.DEBUG


def test_build_arg_parser() -> None:
    """Test argument parser creation."""
    parser = build_arg_parser()
    assert parser is not None

    # Test parsing valid arguments
    args = parser.parse_args(["subjects.csv", "--root", "/path/to/root", "--workers", "4"])
    assert args.csv == Path("subjects.csv")
    assert args.root == Path("/path/to/root")
    assert args.workers == 4


def test_cli_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    """Test CLI dry run mode."""
    # Create CSV file
    csv_path = tmp_path / "subjects.csv"
    csv_path.write_text("subject_code,session_id\n01,baseline\n02,followup\n")

    # Create atlas file
    atlas_path = tmp_path / "atlas.nii.gz"
    atlas_path.touch()

    # Clear any existing env vars
    monkeypatch.delenv("CAT12_ROOT", raising=False)
    monkeypatch.delenv("CAT12_ATLAS_PATHS", raising=False)

    result = main([
        str(csv_path),
        "--root",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "output"),
        "--atlas",
        str(atlas_path),
        "--atlas-name",
        "TestAtlas",
        "--dry-run",
    ])

    assert result == 0
    captured = capsys.readouterr()
    assert "sub-0001_ses-baseline" in captured.out
    assert "sub-0002_ses-followup" in captured.out


def test_cli_missing_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test CLI error when root is not provided."""
    csv_path = tmp_path / "subjects.csv"
    csv_path.write_text("subject_code\n01\n")

    monkeypatch.delenv("CAT12_ROOT", raising=False)
    monkeypatch.setattr("parcellate.interfaces.cat12.cli._load_dotenv", lambda: None)

    result = main([str(csv_path)])

    assert result == 1


def test_cli_missing_atlases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test CLI error when no atlases configured."""
    csv_path = tmp_path / "subjects.csv"
    csv_path.write_text("subject_code\n01\n")

    monkeypatch.delenv("CAT12_ROOT", raising=False)
    monkeypatch.delenv("CAT12_ATLAS_PATHS", raising=False)
    monkeypatch.setattr("parcellate.interfaces.cat12.cli._load_dotenv", lambda: None)

    result = main([str(csv_path), "--root", str(tmp_path)])

    assert result == 1


def test_subject_session_dataclass() -> None:
    """Test SubjectSession dataclass."""
    subj = SubjectSession(subject_code="01", session_id="baseline")
    assert subj.subject_code == "01"
    assert subj.session_id == "baseline"

    subj_no_session = SubjectSession(subject_code="02", session_id=None)
    assert subj_no_session.session_id is None

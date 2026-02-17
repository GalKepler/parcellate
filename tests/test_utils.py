"""Tests for shared interface utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from parcellate.interfaces.utils import (
    _as_list,
    _atlas_threshold_label,
    _mask_label,
    _mask_threshold_label,
    _parse_log_level,
    _parse_mask,
    parse_atlases,
)


class TestParseLogLevel:
    """Tests for _parse_log_level utility."""

    def test_none_returns_info(self) -> None:
        """Test that None returns INFO level."""
        assert _parse_log_level(None) == logging.INFO

    def test_string_debug(self) -> None:
        """Test DEBUG string parsing."""
        assert _parse_log_level("DEBUG") == logging.DEBUG

    def test_string_info(self) -> None:
        """Test INFO string parsing."""
        assert _parse_log_level("INFO") == logging.INFO

    def test_string_warning(self) -> None:
        """Test WARNING string parsing."""
        assert _parse_log_level("WARNING") == logging.WARNING

    def test_string_error(self) -> None:
        """Test ERROR string parsing."""
        assert _parse_log_level("ERROR") == logging.ERROR

    def test_string_case_insensitive(self) -> None:
        """Test case-insensitive parsing."""
        assert _parse_log_level("debug") == logging.DEBUG
        assert _parse_log_level("DeBuG") == logging.DEBUG

    def test_integer_passthrough(self) -> None:
        """Test integer values pass through unchanged."""
        assert _parse_log_level(10) == 10
        assert _parse_log_level(20) == 20
        assert _parse_log_level(logging.DEBUG) == logging.DEBUG

    def test_invalid_string_returns_info(self) -> None:
        """Test invalid string defaults to INFO."""
        assert _parse_log_level("INVALID") == logging.INFO


class TestAsList:
    """Tests for _as_list utility."""

    def test_none_returns_none(self) -> None:
        """Test that None returns None."""
        assert _as_list(None) is None

    def test_single_string_becomes_list(self) -> None:
        """Test single string converted to list."""
        result = _as_list("test")
        assert result == ["test"]
        assert isinstance(result, list)

    def test_list_unchanged(self) -> None:
        """Test list passes through unchanged."""
        input_list = ["a", "b", "c"]
        result = _as_list(input_list)
        assert result == input_list

    def test_tuple_converted_to_list(self) -> None:
        """Test tuple converted to list."""
        result = _as_list(("a", "b"))
        assert result == ["a", "b"]
        assert isinstance(result, list)

    def test_set_converted_to_list(self) -> None:
        """Test set converted to list."""
        result = _as_list({"a", "b"})
        assert isinstance(result, list)
        assert set(result) == {"a", "b"}

    def test_empty_list(self) -> None:
        """Test empty list returns empty list."""
        assert _as_list([]) == []


class TestParseAtlases:
    """Tests for parse_atlases utility."""

    def test_empty_list_returns_empty(self) -> None:
        """Test empty config list returns empty atlas list."""
        assert parse_atlases([]) == []

    def test_single_atlas_minimal(self) -> None:
        """Test parsing single atlas with minimal config."""
        config = [{"name": "TestAtlas", "path": "/path/to/atlas.nii.gz"}]
        atlases = parse_atlases(config)

        assert len(atlases) == 1
        assert atlases[0].name == "TestAtlas"
        assert atlases[0].nifti_path == Path("/path/to/atlas.nii.gz").resolve()
        assert atlases[0].lut is None
        assert atlases[0].space is None
        assert atlases[0].resolution is None

    def test_single_atlas_complete(self) -> None:
        """Test parsing single atlas with all fields."""
        config = [
            {
                "name": "Schaefer400",
                "path": "/path/to/schaefer.nii.gz",
                "lut": "/path/to/schaefer.tsv",
                "space": "MNI152NLin2009cAsym",
                "resolution": "2mm",
            }
        ]
        atlases = parse_atlases(config)

        assert len(atlases) == 1
        atlas = atlases[0]
        assert atlas.name == "Schaefer400"
        assert atlas.nifti_path == Path("/path/to/schaefer.nii.gz").resolve()
        assert atlas.lut == Path("/path/to/schaefer.tsv").resolve()
        assert atlas.space == "MNI152NLin2009cAsym"
        assert atlas.resolution == "2mm"

    def test_multiple_atlases(self) -> None:
        """Test parsing multiple atlases."""
        config = [
            {"name": "Atlas1", "path": "/path/1.nii.gz"},
            {"name": "Atlas2", "path": "/path/2.nii.gz"},
            {"name": "Atlas3", "path": "/path/3.nii.gz"},
        ]
        atlases = parse_atlases(config)

        assert len(atlases) == 3
        assert [a.name for a in atlases] == ["Atlas1", "Atlas2", "Atlas3"]

    def test_default_space_applied(self) -> None:
        """Test default_space applied when space not in config."""
        config = [{"name": "Test", "path": "/path/test.nii.gz"}]
        atlases = parse_atlases(config, default_space="MNI152NLin2009cAsym")

        assert atlases[0].space == "MNI152NLin2009cAsym"

    def test_explicit_space_overrides_default(self) -> None:
        """Test explicit space in config overrides default."""
        config = [{"name": "Test", "path": "/path/test.nii.gz", "space": "T1w"}]
        atlases = parse_atlases(config, default_space="MNI152NLin2009cAsym")

        assert atlases[0].space == "T1w"

    def test_skips_missing_name(self) -> None:
        """Test atlas with missing name is skipped."""
        config = [
            {"name": "Valid", "path": "/path/valid.nii.gz"},
            {"path": "/path/no_name.nii.gz"},
        ]
        atlases = parse_atlases(config)

        assert len(atlases) == 1
        assert atlases[0].name == "Valid"

    def test_skips_missing_path(self) -> None:
        """Test atlas with missing path is skipped."""
        config = [
            {"name": "Valid", "path": "/path/valid.nii.gz"},
            {"name": "NoPath"},
        ]
        atlases = parse_atlases(config)

        assert len(atlases) == 1
        assert atlases[0].name == "Valid"

    def test_skips_empty_name(self) -> None:
        """Test atlas with empty name is skipped."""
        config = [
            {"name": "Valid", "path": "/path/valid.nii.gz"},
            {"name": "", "path": "/path/empty_name.nii.gz"},
        ]
        atlases = parse_atlases(config)

        assert len(atlases) == 1
        assert atlases[0].name == "Valid"

    def test_skips_empty_path(self) -> None:
        """Test atlas with empty path is skipped."""
        config = [
            {"name": "Valid", "path": "/path/valid.nii.gz"},
            {"name": "EmptyPath", "path": ""},
        ]
        atlases = parse_atlases(config)

        assert len(atlases) == 1
        assert atlases[0].name == "Valid"

    def test_path_expansion(self) -> None:
        """Test tilde expansion in paths."""
        config = [{"name": "Test", "path": "~/atlas.nii.gz"}]
        atlases = parse_atlases(config)

        assert "~" not in str(atlases[0].nifti_path)
        assert atlases[0].nifti_path.is_absolute()

    def test_lut_path_expansion(self) -> None:
        """Test tilde expansion in LUT paths."""
        config = [{"name": "Test", "path": "/path/atlas.nii.gz", "lut": "~/lut.tsv"}]
        atlases = parse_atlases(config)

        assert atlases[0].lut is not None
        assert "~" not in str(atlases[0].lut)
        assert atlases[0].lut.is_absolute()

    def test_preserves_order(self) -> None:
        """Test atlases are returned in config order."""
        config = [
            {"name": "First", "path": "/1.nii.gz"},
            {"name": "Second", "path": "/2.nii.gz"},
            {"name": "Third", "path": "/3.nii.gz"},
        ]
        atlases = parse_atlases(config)

        names = [a.name for a in atlases]
        assert names == ["First", "Second", "Third"]

    def test_all_invalid_returns_empty(self) -> None:
        """Test all invalid configs returns empty list."""
        config = [
            {"name": "NoPath"},
            {"path": "/no_name.nii.gz"},
            {"name": "", "path": ""},
        ]
        atlases = parse_atlases(config)

        assert atlases == []


class TestParseMask:
    """Tests for _parse_mask utility."""

    def test_none_returns_none(self) -> None:
        """Test that None returns None."""
        assert _parse_mask(None) is None

    def test_empty_string_returns_none(self) -> None:
        """Test that empty string returns None."""
        assert _parse_mask("") is None

    def test_gm_returns_string(self) -> None:
        """Test that 'gm' builtin name is returned as a plain string."""
        result = _parse_mask("gm")
        assert result == "gm"
        assert isinstance(result, str)

    def test_wm_returns_string(self) -> None:
        """Test that 'wm' builtin name is returned as a plain string."""
        result = _parse_mask("wm")
        assert result == "wm"
        assert isinstance(result, str)

    def test_brain_returns_string(self) -> None:
        """Test that 'brain' builtin name is returned as a plain string."""
        result = _parse_mask("brain")
        assert result == "brain"
        assert isinstance(result, str)

    def test_path_string_returns_resolved_path(self) -> None:
        """Test that a filesystem path string is returned as a resolved Path."""
        result = _parse_mask("/var/mask.nii.gz")
        assert isinstance(result, Path)
        assert result == Path("/var/mask.nii.gz").resolve()

    def test_relative_path_is_resolved(self) -> None:
        """Test that relative paths are resolved to absolute."""
        result = _parse_mask("some/relative/mask.nii.gz")
        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_tilde_path_is_expanded(self) -> None:
        """Test that tilde is expanded in path strings."""
        result = _parse_mask("~/mask.nii.gz")
        assert isinstance(result, Path)
        assert "~" not in str(result)
        assert result.is_absolute()

    def test_non_builtin_name_treated_as_path(self) -> None:
        """Test that unrecognized names are treated as paths, not builtin strings."""
        result = _parse_mask("csf")
        assert isinstance(result, Path)


class TestMaskLabel:
    """Tests for _mask_label utility."""

    def test_none_returns_none(self) -> None:
        """Test that None produces no entity label."""
        assert _mask_label(None) is None

    def test_builtin_string_returns_name(self) -> None:
        """Test that builtin name strings are returned as-is."""
        assert _mask_label("gm") == "gm"
        assert _mask_label("wm") == "wm"
        assert _mask_label("brain") == "brain"

    def test_path_returns_custom(self) -> None:
        """Test that any Path produces the 'custom' label."""
        assert _mask_label(Path("/some/mask.nii.gz")) == "custom"
        assert _mask_label(Path("relative/mask.nii.gz")) == "custom"


class TestMaskThresholdLabel:
    """Tests for _mask_threshold_label utility."""

    def test_zero_returns_none(self) -> None:
        """Default threshold 0.0 produces no entity label."""
        assert _mask_threshold_label(0.0) is None

    def test_clean_fraction_returns_percentage(self) -> None:
        """Clean fractions are expressed as integer percentages."""
        assert _mask_threshold_label(0.5) == "50"
        assert _mask_threshold_label(0.25) == "25"
        assert _mask_threshold_label(0.1) == "10"

    def test_non_clean_fraction_uses_p_separator(self) -> None:
        """Non-clean fractions use 'p' instead of '.'."""
        # 1/3 cannot be expressed as a clean integer percentage
        val = 1 / 3
        result = _mask_threshold_label(val)
        assert result is not None
        assert "p" in result or "." not in result  # uses 'p' as decimal separator


class TestAtlasThresholdLabel:
    """Tests for _atlas_threshold_label utility."""

    def test_zero_returns_none(self) -> None:
        """Default threshold 0.0 produces no entity label."""
        assert _atlas_threshold_label(0.0) is None

    def test_clean_fraction_returns_percentage(self) -> None:
        """Clean fractions are expressed as integer percentages."""
        assert _atlas_threshold_label(0.5) == "50"
        assert _atlas_threshold_label(0.25) == "25"
        assert _atlas_threshold_label(0.1) == "10"

    def test_non_clean_fraction_uses_p_separator(self) -> None:
        """Non-clean fractions use 'p' instead of '.'."""
        # 1/3 cannot be expressed as a clean integer percentage
        val = 1 / 3
        result = _atlas_threshold_label(val)
        assert result is not None
        assert "." not in result  # '.' is replaced with 'p'

    def test_one_returns_100(self) -> None:
        """Threshold of 1.0 returns '100'."""
        assert _atlas_threshold_label(1.0) == "100"


class TestParseAtlasesThreshold:
    """Tests for atlas_threshold in parse_atlases."""

    def test_default_atlas_threshold_is_zero(self) -> None:
        """Atlas with no atlas_threshold field defaults to 0.0."""
        config = [{"name": "Test", "path": "/path/atlas.nii.gz"}]
        atlases = parse_atlases(config)

        assert atlases[0].atlas_threshold == 0.0

    def test_atlas_threshold_parsed(self) -> None:
        """atlas_threshold is read from config dict."""
        config = [{"name": "XTRACT", "path": "/path/xtract.nii.gz", "atlas_threshold": 0.25}]
        atlases = parse_atlases(config)

        assert atlases[0].atlas_threshold == pytest.approx(0.25)

    def test_atlas_threshold_zero_explicit(self) -> None:
        """Explicit atlas_threshold=0.0 is correctly stored."""
        config = [{"name": "Test", "path": "/path/atlas.nii.gz", "atlas_threshold": 0.0}]
        atlases = parse_atlases(config)

        assert atlases[0].atlas_threshold == 0.0

    def test_multiple_atlases_independent_thresholds(self) -> None:
        """Each atlas can have its own independent atlas_threshold."""
        config = [
            {"name": "Atlas1", "path": "/path/1.nii.gz", "atlas_threshold": 0.1},
            {"name": "Atlas2", "path": "/path/2.nii.gz", "atlas_threshold": 0.5},
            {"name": "Atlas3", "path": "/path/3.nii.gz"},
        ]
        atlases = parse_atlases(config)

        assert atlases[0].atlas_threshold == pytest.approx(0.1)
        assert atlases[1].atlas_threshold == pytest.approx(0.5)
        assert atlases[2].atlas_threshold == 0.0

"""Tests for main CLI entry point."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from parcellate.cli import main


class TestMainCLI:
    """Tests for the unified CLI dispatcher."""

    def test_no_args_prints_help(self, capsys) -> None:
        """Test running with no args prints help and returns 1."""
        result = main([])
        captured = capsys.readouterr()

        assert result == 1
        assert "parcellate" in captured.out.lower()
        assert "cat12" in captured.out.lower() or "qsirecon" in captured.out.lower()

    def test_help_flag(self, capsys) -> None:
        """Test --help flag displays help."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "parcellate" in captured.out.lower()

    def test_cat12_subcommand_delegates(self) -> None:
        """Test cat12 subcommand delegates to cat12 module."""
        with patch("parcellate.interfaces.cat12.cat12.main") as mock_cat12_main:
            mock_cat12_main.return_value = 0

            result = main(["cat12", "config.toml"])

            assert result == 0
            mock_cat12_main.assert_called_once()
            # The cat12 subcommand uses positional 'config' arg, so remaining will be empty
            # after argparse extracts it

    @patch("parcellate.cli.sys")
    def test_qsirecon_subcommand_delegates(self, mock_sys) -> None:
        """Test qsirecon subcommand delegates to qsirecon module."""
        with patch("parcellate.interfaces.qsirecon.qsirecon.main") as mock_qsi_main:
            mock_qsi_main.return_value = 0

            result = main(["qsirecon", "--config", "config.toml"])

            assert result == 0
            mock_qsi_main.assert_called_once()

    def test_cat12_with_config_file(self) -> None:
        """Test cat12 subcommand accepts config argument."""
        with patch("parcellate.interfaces.cat12.cat12.main") as mock_main:
            mock_main.return_value = 0

            result = main(["cat12", "/path/to/config.toml"])

            assert result == 0
            mock_main.assert_called_once()

    def test_qsirecon_with_multiple_args(self) -> None:
        """Test qsirecon accepts multiple arguments."""
        with patch("parcellate.interfaces.qsirecon.qsirecon.main") as mock_main:
            mock_main.return_value = 0

            result = main([
                "qsirecon",
                "--config",
                "config.toml",
                "--input-root",
                "/data",
            ])

            assert result == 0
            mock_main.assert_called_once()

    def test_cat12_error_propagates(self) -> None:
        """Test error from cat12 module propagates."""
        with patch("parcellate.interfaces.cat12.cat12.main") as mock_main:
            mock_main.return_value = 1

            result = main(["cat12", "config.toml"])

            assert result == 1

    def test_qsirecon_error_propagates(self) -> None:
        """Test error from qsirecon module propagates."""
        with patch("parcellate.interfaces.qsirecon.qsirecon.main") as mock_main:
            mock_main.return_value = 1

            result = main(["qsirecon", "--config", "config.toml"])

            assert result == 1

    def test_no_pipeline_flag_returns_error(self, capsys) -> None:
        """Test missing --pipeline flag raises SystemExit with error code 2."""
        # When bids_dir/output_dir/analysis_level are provided but --pipeline is missing,
        # argparse exits with code 2.
        with pytest.raises(SystemExit) as exc_info:
            main(["/bids", "/out", "participant"])

        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "--pipeline" in captured.err.lower() or "required" in captured.err.lower()

    def test_cat12_preserves_arguments(self) -> None:
        """Test cat12 subcommand calls main function."""
        with patch("parcellate.interfaces.cat12.cat12.main") as mock_main:
            mock_main.return_value = 0
            test_args = ["cat12", "my_config.toml"]

            result = main(test_args)

            # Should call cat12.main
            assert result == 0
            mock_main.assert_called_once()

    def test_qsirecon_preserves_arguments(self) -> None:
        """Test qsirecon subcommand preserves all arguments."""
        with patch("parcellate.interfaces.qsirecon.qsirecon.main") as mock_main:
            mock_main.return_value = 0
            test_args = [
                "qsirecon",
                "--config",
                "config.toml",
                "--subjects",
                "sub-01",
                "sub-02",
            ]

            main(test_args)

            # Should pass all args after 'qsirecon'
            passed_args = mock_main.call_args[0][0]
            assert "--config" in passed_args
            assert "config.toml" in passed_args


class TestBIDSAppCLI:
    """Tests for the new BIDS App-style CLI interface."""

    def test_bids_app_cat12_dispatches_correctly(self, tmp_path) -> None:
        """Test BIDS App positional args with --pipeline cat12 dispatches to cat12."""
        bids_dir = tmp_path / "bids"
        out_dir = tmp_path / "out"
        bids_dir.mkdir()

        with (
            patch("parcellate.interfaces.cat12.cat12.load_config") as mock_config,
            patch("parcellate.interfaces.cat12.cat12.run_parcellations") as mock_run,
        ):
            mock_run.return_value = []
            result = main([str(bids_dir), str(out_dir), "participant", "--pipeline", "cat12"])

        assert result == 0
        mock_config.assert_called_once()
        mock_run.assert_called_once()

    def test_bids_app_qsirecon_dispatches_correctly(self, tmp_path) -> None:
        """Test BIDS App positional args with --pipeline qsirecon dispatches to qsirecon."""
        bids_dir = tmp_path / "bids"
        out_dir = tmp_path / "out"
        bids_dir.mkdir()

        with (
            patch("parcellate.interfaces.qsirecon.qsirecon.load_config") as mock_config,
            patch("parcellate.interfaces.qsirecon.qsirecon.run_parcellations") as mock_run,
        ):
            mock_run.return_value = []
            result = main([str(bids_dir), str(out_dir), "participant", "--pipeline", "qsirecon"])

        assert result == 0
        mock_config.assert_called_once()
        mock_run.assert_called_once()

    def test_bids_app_participant_labels_passed(self, tmp_path) -> None:
        """Test --participant-label translates to subjects in config namespace."""
        from parcellate.cli import _build_bids_parser, _to_pipeline_namespace

        parser = _build_bids_parser()
        args = parser.parse_args([
            str(tmp_path),
            str(tmp_path),
            "participant",
            "--pipeline",
            "cat12",
            "--participant-label",
            "01",
            "02",
        ])
        pipeline_ns = _to_pipeline_namespace(args)
        assert pipeline_ns.subjects == ["01", "02"]

    def test_bids_app_session_label_passed(self, tmp_path) -> None:
        """Test --session-label translates to sessions in config namespace."""
        from parcellate.cli import _build_bids_parser, _to_pipeline_namespace

        parser = _build_bids_parser()
        args = parser.parse_args([
            str(tmp_path),
            str(tmp_path),
            "participant",
            "--pipeline",
            "cat12",
            "--session-label",
            "ses-01",
        ])
        pipeline_ns = _to_pipeline_namespace(args)
        assert pipeline_ns.sessions == ["ses-01"]

    def test_bids_app_bids_dir_maps_to_input_root(self, tmp_path) -> None:
        """Test bids_dir positional arg maps to input_root in pipeline namespace."""
        from parcellate.cli import _build_bids_parser, _to_pipeline_namespace

        parser = _build_bids_parser()
        args = parser.parse_args([str(tmp_path), str(tmp_path), "participant", "--pipeline", "cat12"])
        pipeline_ns = _to_pipeline_namespace(args)
        assert pipeline_ns.input_root == tmp_path

    def test_legacy_subcommand_emits_deprecation_warning(self) -> None:
        """Test that legacy subcommand mode emits a DeprecationWarning."""
        with (
            patch("parcellate.interfaces.cat12.cat12.main") as mock_main,
            pytest.warns(DeprecationWarning, match="deprecated"),
        ):
            mock_main.return_value = 0
            main(["cat12", "--help"])

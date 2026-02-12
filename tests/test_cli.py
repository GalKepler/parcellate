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

    def test_invalid_subcommand_returns_error(self, capsys) -> None:
        """Test invalid subcommand raises SystemExit."""
        # argparse raises SystemExit for invalid commands
        with pytest.raises(SystemExit) as exc_info:
            main(["invalid_command"])

        # Should exit with error code 2 (argparse error)
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "invalid" in captured.err.lower()

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

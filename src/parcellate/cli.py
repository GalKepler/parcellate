"""Unified CLI entry point for parcellate.

Usage::

    parcellate cat12 config.toml
    parcellate qsirecon config.toml
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="parcellate",
        description="Extract regional statistics from scalar neuroimaging maps.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- cat12 subcommand ---
    _ = subparsers.add_parser(
        "cat12",
        help="Run parcellations for CAT12 derivatives.",
        add_help=False,
    )

    # --- qsirecon subcommand ---
    _ = subparsers.add_parser(
        "qsirecon",
        help="Run parcellations for QSIRecon derivatives.",
        add_help=False,
    )

    args, remaining = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "cat12":
        from parcellate.interfaces.cat12.cat12 import main as cat12_main

        return cat12_main(remaining)

    if args.command == "qsirecon":
        from parcellate.interfaces.qsirecon.qsirecon import main as qsirecon_main

        return qsirecon_main(remaining)

    # Unreachable when argparse is configured correctly, but keeps mypy happy.
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

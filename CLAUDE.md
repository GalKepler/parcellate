# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`parcellate` is a neuroimaging tool for extracting regional statistics from scalar brain maps using atlas-based parcellation. It integrates with CAT12 and QSIRecon preprocessing pipelines and outputs BIDS-derivative-style results.

## Development Commands

```bash
# Environment
make install              # Create venv with uv + install pre-commit hooks
uv sync                   # Sync dependencies

# Code quality
make check                # Run all checks (lock file, pre-commit, deptry)
uv run pre-commit run -a  # Run pre-commit hooks (ruff format + ruff lint)
uv run mypy src           # Type checking (NOT part of pre-commit, run separately)
uv run deptry src         # Check for obsolete dependencies

# Testing
make test                              # All tests with coverage
uv run python -m pytest                # All tests without coverage
uv run python -m pytest tests/test_parcellator.py  # Specific file
uv run python -m pytest tests/test_parcellator.py::test_name  # Specific test
uv run python -m pytest -k "pattern"   # Tests matching pattern

# Docs
make docs       # Build and serve docs locally
make docs-test  # Test docs build
```

## High-Level Architecture

### Core Engine

**VolumetricParcellator** (`src/parcellate/parcellation/volume.py`) - the main parcellation engine:
- `.fit(scalar_map)` loads/resamples the scalar map; `.transform(scalar_map)` extracts regional statistics
- Caches fitted scalar images to avoid redundant resampling when fit/transform use the same image
- Defaults to `resampling_target="data"`, resampling maps to atlas space to preserve atlas label boundaries

**Metrics** (`src/parcellate/metrics/`) - `Statistic` protocol in `base.py`, built-in stats (mean, std, median, volume, etc.) in `volume.py`. Custom statistics via `stat_functions` parameter.

### Interface Architecture

Each preprocessing pipeline (CAT12, QSIRecon) has its own interface under `src/parcellate/interfaces/<pipeline>/`, but they share core orchestration:

- **Shared**: `models.py` (data structures), `planner.py` (matches atlases to scalar maps by space), `runner.py` (runs parcellation), `utils.py`
- **Per-interface**: `loader.py` (discovers pipeline outputs), `models.py` (pipeline-specific config/definitions)

The shared planner/runner use duck typing - any config with `mask`, `background_label`, and `resampling_target` attributes works.

### Data Flow

1. **Loader** discovers subject data (scalar maps, transforms, atlases) from preprocessing outputs
2. **Planner** matches atlases to scalar maps based on space compatibility
3. **Runner** instantiates VolumetricParcellator for each atlas, runs `.fit()` + `.transform()`
4. **Output** saves parcellation stats as TSV files in BIDS-derivative structure

### CLI Entry Points

Two distinct CLI patterns exist:

```bash
# TOML-based (unified CLI)
parcellate cat12 config.toml       # Routes through src/parcellate/cli.py -> cat12.py
parcellate qsirecon config.toml    # Routes through src/parcellate/cli.py -> qsirecon.py

# CSV + environment-variable-based (standalone CLI)
parcellate-cat12 subjects.csv      # src/parcellate/interfaces/cat12/cli.py
```

The `parcellate-cat12` CLI is different from `parcellate cat12` - it takes a CSV file (with `subject_code`/`session_id` columns) and reads atlas/path configuration from environment variables (`CAT12_ROOT`, `CAT12_ATLAS_PATHS`, etc.) or CLI flags. Supports `.env` files via optional `python-dotenv`.

## Configuration

TOML configuration files control the unified CLI workflows:

```toml
input_root = "/path/to/derivatives"
output_dir = "/path/to/parcellations"
subjects = ["sub-01", "sub-02"]  # optional filter
sessions = ["ses-01"]            # optional filter
mask = "/path/to/mask.nii.gz"   # optional brain mask
force = false                    # overwrite existing outputs
log_level = "INFO"
n_jobs = 4                       # parallel jobs within-subject
n_procs = 2                      # parallel processes across-subjects

[[atlases]]
name = "schaefer400"
path = "/path/to/atlas.nii.gz"
lut = "/path/to/atlas.tsv"       # optional lookup table
space = "MNI152NLin6Asym"
```

## Code Quality Standards

- **Ruff**: line length 120, flake8-style rules. `E501` (line too long) and `E731` (lambda assignment) are ignored. Tests allow `S101` (assert).
- **Mypy**: strict settings (`disallow_untyped_defs`, `check_untyped_defs`). Run manually, not in pre-commit.
- **Python**: targets 3.9+ (`from __future__ import annotations` used throughout for modern type syntax)

## Adding New Pipeline Interfaces

1. Create `src/parcellate/interfaces/newpipeline/` directory
2. Implement `loader.py` to discover pipeline outputs
3. Define pipeline-specific models in `models.py` (extend `ScalarMapBase`)
4. Create `newpipeline.py` with TOML parser and main() function
5. Add subcommand to `src/parcellate/cli.py`
6. Reuse shared `planner.py` and `runner.py` - no modifications needed

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`parcellate` is a neuroimaging tool for extracting regional statistics from scalar brain maps using atlas-based parcellation. It integrates with CAT12 and QSIRecon preprocessing pipelines and outputs BIDS-derivative-style results.

## Development Commands

### Environment Setup
```bash
make install              # Create virtual environment with uv and install pre-commit hooks
uv sync                   # Sync dependencies (alternative to make install)
```

### Code Quality
```bash
make check                # Run all quality checks (lock file, pre-commit, deptry)
uv run pre-commit run -a  # Run pre-commit hooks (ruff format, ruff lint, mypy)
uv run deptry src         # Check for obsolete dependencies
```

### Testing
```bash
make test                              # Run all tests with coverage
uv run python -m pytest                # Run tests without coverage
uv run python -m pytest tests/test_parcellator.py  # Run specific test file
uv run python -m pytest tests/test_parcellator.py::test_name  # Run specific test
uv run python -m pytest -v            # Verbose test output
uv run python -m pytest -k "pattern"  # Run tests matching pattern
```

### Documentation
```bash
make docs       # Build and serve documentation locally
make docs-test  # Test if documentation builds without errors
```

### Building & Publishing
```bash
make build              # Build wheel file
make build-and-publish  # Build and publish to PyPI (requires PYPI_TOKEN)
```

## High-Level Architecture

### Core Components

**VolumetricParcellator** (`src/parcellate/parcellation/volume.py`)
- Main parcellation engine that samples scalar maps using an integer-valued atlas
- `.fit(scalar_map)` - loads and prepares the scalar map for parcellation
- `.transform(scalar_map)` - extracts regional statistics (mean, std, volume, etc.)
- Handles resampling between atlas and data spaces via nilearn
- Caches fitted scalar images to avoid redundant resampling when the same image is used in fit() and transform()

**Metrics System** (`src/parcellate/metrics/`)
- `base.py` - defines the `Statistic` protocol for aggregation functions
- `volume.py` - provides `BUILTIN_STATISTICS` (mean, std, median, volume, etc.)
- Custom statistics can be added via `stat_functions` parameter

### Interface Architecture

The package supports multiple preprocessing pipelines through a shared interface design:

**Shared Components** (`src/parcellate/interfaces/`)
- `models.py` - common data structures: `SubjectContext`, `AtlasDefinition`, `ReconInput`, `ParcellationOutput`
- `planner.py` - `plan_parcellation_workflow()` matches atlases to scalar maps by space
- `runner.py` - `run_parcellation_workflow()` orchestrates parcellation jobs
- `utils.py` - helper functions for config parsing

**Interface-Specific Modules**
- `cat12/` - CAT12 VBM pipeline integration
  - `loader.py` - discovers CAT12 outputs (GM/WM/CSF volumes, thickness maps)
  - `cat12.py` - TOML config parser and parallel orchestration
  - `cli.py` - dedicated CLI entry point: `parcellate-cat12`
  - `models.py` - CAT12-specific definitions (`TissueType`, `ScalarMapDefinition`, `Cat12Config`)

- `qsirecon/` - QSIRecon diffusion pipeline integration
  - `loader.py` - discovers QSIRecon outputs (FA, MD, etc.)
  - `qsirecon.py` - TOML config parser
  - `models.py` - QSIRecon-specific definitions (`ScalarMapDefinition`, `QSIReconConfig`)

**Design Pattern**: Each interface defines its own discovery logic (`loader.py`) and config schema (`models.py`), but shares the core orchestration logic (planner/runner). This allows adding new pipelines without duplicating workflow code.

### CLI Structure

```bash
parcellate cat12 config.toml      # Main CLI for CAT12
parcellate qsirecon config.toml   # Main CLI for QSIRecon
parcellate-cat12 config.toml      # Alternative CAT12 entry point
```

- `src/parcellate/cli.py` - unified entry point routing to interface-specific handlers
- Each interface CLI parses TOML config, runs loader, planner, runner, and saves outputs

### Data Flow

1. **Loader** discovers subject data (scalar maps, transforms, atlases) from preprocessing outputs
2. **Planner** matches atlases to scalar maps based on space compatibility
3. **Runner** instantiates VolumetricParcellator for each atlas and runs `.fit()` + `.transform()`
4. **Output** saves parcellation statistics as TSV files in BIDS-derivative structure

## Configuration

TOML configuration files control parcellation workflows:

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

## Project Structure

```
src/parcellate/
├── cli.py                    # Unified CLI entry point
├── parcellation/
│   └── volume.py            # VolumetricParcellator core engine
├── metrics/
│   ├── base.py              # Statistic protocol
│   └── volume.py            # Built-in aggregation functions
├── interfaces/              # Pipeline integrations
│   ├── models.py            # Shared data structures
│   ├── planner.py           # Workflow planning
│   ├── runner.py            # Workflow execution
│   ├── utils.py             # Config parsing helpers
│   ├── cat12/               # CAT12 interface
│   └── qsirecon/            # QSIRecon interface
└── utils/
    └── image.py             # NIfTI loading utilities

tests/
├── test_parcellator.py      # VolumetricParcellator tests
├── test_metrics.py          # Statistics function tests
├── test_interfaces_cat12.py # CAT12 interface tests
└── test_interfaces_qsirecon.py  # QSIRecon interface tests
```

## Testing Strategy

- **Unit tests** for parcellator, metrics, and utilities use synthetic NIfTI images
- **Interface tests** mock file discovery and validate workflow orchestration
- **Fixtures** in `conftest.py` provide reusable test data
- All tests run via pytest with coverage reporting to codecov

## Code Quality Standards

- **Linting**: ruff with flake8-style rules (line length 120, PEP8 compliance)
- **Type checking**: mypy with strict settings (`disallow_untyped_defs`, `check_untyped_defs`)
- **Testing**: pytest with coverage requirements
- **Pre-commit hooks**: Auto-format and lint on commit

## Recent Refactoring Context

The codebase recently underwent three phases of refactoring:
1. **Phase 1** - Fixed `voxel_count()` parameter naming and documentation
2. **Phase 2** - Added scalar image caching to avoid redundant resampling
3. **Phase 3** - Optimized statistics loop with dict-based accumulation

See `refactoring_plan.md` and `plans/` directory for detailed refactoring history and architecture evolution.

## Adding New Pipeline Interfaces

To add a new preprocessing pipeline (e.g., fMRIPrep):

1. Create `src/parcellate/interfaces/newpipeline/` directory
2. Implement `loader.py` to discover pipeline outputs
3. Define pipeline-specific models in `models.py` (extend `ScalarMapBase`)
4. Create `newpipeline.py` with TOML parser and main() function
5. Add subcommand to `src/parcellate/cli.py`
6. Reuse shared `planner.py` and `runner.py` - no modifications needed

The shared planner/runner use duck typing to accept any config with `mask`, `background_label`, and `resampling_target` attributes.

## Important Implementation Notes

- **Resampling**: VolumetricParcellator defaults to `resampling_target="data"`, resampling input maps to atlas space to preserve atlas boundaries
- **Caching**: After `.fit()`, the parcellator caches the prepared scalar image and reuses it in `.transform()` if the same path/object is provided
- **Space matching**: The planner validates that atlas and scalar map spaces are compatible before creating parcellation jobs
- **Parallel execution**: CAT12 interface supports both within-subject parallelization (`n_jobs`) and across-subject parallelization (`n_procs`)

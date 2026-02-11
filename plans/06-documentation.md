# Plan 06: Documentation Improvements

## 6.1 README Overhaul

**File:** `README.md` (currently 78 lines, mostly cookiecutter template)

### Current Issues
- Title and description are minimal
- "Getting started" section is generic boilerplate
- No usage examples
- No description of what the tool actually does
- No explanation of CAT12 vs QSIRecon pipelines

### Proposed Structure

```markdown
# parcellate

> Extract regional statistics from scalar neuroimaging maps using atlas-based parcellation.

## What It Does
- Brief explanation of volumetric parcellation
- Supported input formats (CAT12, QSIRecon)
- Output format (BIDS-derivative TSV tables)

## Installation
  pip install parcellate
  pip install parcellate[dotenv]  # for .env support in CAT12 CSV mode

## Quick Start

### CAT12 Pipeline
  parcellate cat12 config.toml
- Show example TOML config
- Show example output

### QSIRecon Pipeline
  parcellate qsirecon --config config.toml --input-root /path/to/qsirecon
- Show example TOML config

### CAT12 CSV Mode (Batch Processing)
  parcellate-cat12 subjects.csv --root /path/to/cat12
- Show example CSV file

## Configuration Reference
- Table of all TOML fields
- Table of environment variables (CAT12 CSV mode)
- Atlas definition format

## Output Format
- TSV column descriptions
- Example output table

## Available Statistics
- Table of all 13+ builtin statistics with descriptions

## Python API
  from parcellate import VolumetricParcellator
- Minimal code example

## Development
- Contributing guide link
- How to run tests
```

---

## 6.2 Module-Level Docstrings

Several `__init__.py` files lack module docstrings:

| File | Status |
|------|--------|
| `parcellate/__init__.py` | Needs docstring |
| `interfaces/__init__.py` | Empty, needs docstring |
| `metrics/__init__.py` | Needs docstring |
| `parcellation/__init__.py` | Needs docstring |
| `utils/__init__.py` | Has exports but no docstring |

### Action

Add one-line module docstrings describing the subpackage purpose:

```python
# parcellate/__init__.py
"""Atlas-based volumetric parcellation of scalar neuroimaging maps."""

# interfaces/__init__.py
"""Interface modules for loading data from neuroimaging pipelines."""

# metrics/__init__.py
"""Statistical metrics for summarizing parcellated regions."""
```

---

## 6.3 Configuration Examples

### TOML Config Documentation

Create example config files or document the format inline:

**CAT12 example (`examples/cat12_config.toml`):**
```toml
input_root = "/data/cat12_derivatives"
output_dir = "/data/parcellations"
force = false
log_level = "INFO"
n_jobs = 4
n_procs = 2
mask = "gm"

[[atlases]]
name = "Schaefer400"
path = "/atlases/Schaefer400_7Networks.nii.gz"
lut = "/atlases/Schaefer400_7Networks.tsv"
space = "MNI152NLin2009cAsym"

[[atlases]]
name = "AAL3"
path = "/atlases/AAL3v1.nii.gz"
lut = "/atlases/AAL3v1.tsv"
space = "MNI152NLin2009cAsym"
```

**QSIRecon example (`examples/qsirecon_config.toml`):**
```toml
input_root = "/data/qsirecon_derivatives"
output_dir = "/data/parcellations"
subjects = ["sub-01", "sub-02"]
n_jobs = 4

[[atlases]]
name = "BNA"
path = "/atlases/BNA.nii.gz"
lut = "/atlases/BNA.tsv"
space = "MNI152NLin2009cAsym"
```

---

## 6.4 Docstring Gaps

Most functions have good docstrings. Specific gaps:

| Function | File | Issue |
|----------|------|-------|
| `_space_match()` | `interfaces/planner.py:14` | Docstring is one line, no parameter docs |
| `_parcellate_scalar_map()` | `interfaces/runner.py:62` | Missing Parameters section |
| `_load_atlas_data()` | `parcellation/volume.py:232` | No docstring at all |
| `_scalar_name()` | `qsirecon/loader.py:251` | Missing Returns section |
| `_workflow_name()` | `qsirecon/loader.py:262` | Missing Returns section |
| `_find_lut_file()` | `qsirecon/loader.py:229` | Parameters section incomplete |

---

## 6.5 MkDocs Site Expansion

**Current state:** MkDocs is configured with mkdocstrings for auto-generated API docs.

### Additions

1. **User Guide page** - Step-by-step walkthrough for each pipeline
2. **Configuration Reference page** - Complete TOML schema documentation
3. **Output Format page** - Column definitions for TSV output
4. **Statistics Reference page** - Description of each builtin statistic, formula, and edge case behavior
5. **FAQ / Troubleshooting page** - Common errors and solutions

---

## 6.6 Inline Code Comments

A few complex sections could benefit from brief comments:

| Location | Suggestion |
|----------|-----------|
| `volume.py:294-309` (fit) | Comment explaining resampling target logic |
| `volume.py:323-336` (transform) | Comment explaining scalar image caching logic |
| `volume.py:345-368` (transform) | Comment explaining the stats accumulation strategy |
| `runner.py:146` (fit first scalar) | Comment explaining why the first scalar map is used for fitting |
| `planner.py:38` | Comment explaining space matching strategy |

---

## Summary

| Item | Effort | Priority |
|------|--------|----------|
| README overhaul | Medium | High |
| Module docstrings | Small | Medium |
| Configuration examples | Small | Medium |
| Function docstring gaps | Small | Low |
| MkDocs expansion | Large | Low |
| Inline comments | Small | Low |

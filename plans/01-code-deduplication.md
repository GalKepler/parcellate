# Plan 01: Code Deduplication

## Problem

The CAT12 and QSIRecon interface modules have significant code duplication (~15-20% of interface code). This makes maintenance harder and increases the risk of bugs diverging between the two pipelines.

## Duplicated Code Inventory

### 1. `_parse_atlases()` - Identical logic

**Files:**
- `interfaces/cat12/cat12.py:86-120`
- `interfaces/qsirecon/qsirecon.py:86-120`

Both functions parse `list[dict]` into `list[AtlasDefinition]` with identical validation, path resolution, and construction. The only difference is the default space value in CAT12 (`"MNI152NLin2009cAsym"` on line 108 of cat12.py) vs no default in QSIRecon.

**Action:** Extract to `interfaces/utils.py` as a shared function with an optional `default_space` parameter.

```python
# interfaces/utils.py
def parse_atlases(
    atlas_configs: list[dict],
    default_space: str | None = None,
) -> list[AtlasDefinition]:
```

### 2. `_build_output_path()` - Near-identical structure

**Files:**
- `interfaces/cat12/cat12.py:123-150`
- `interfaces/qsirecon/qsirecon.py:123-154`
- `interfaces/cat12/cli.py:227-253` (third copy!)

All three build a BIDS-derivative path from `SubjectContext + AtlasDefinition + ScalarMapDefinition`. Differences:
- CAT12: `base = destination / "cat12"`, modality dir is `"anat"`, uses `tissue_type` entity
- QSIRecon: `base = destination / f"qsirecon-{workflow}"`, modality dir is `"dwi"`, uses `model`, `param`, `desc` entities

**Action:** Extract to a shared function with a "profile" or callback for interface-specific entity construction:

```python
# interfaces/utils.py
def build_output_path(
    context: SubjectContext,
    atlas: AtlasDefinition,
    scalar_map: Any,
    destination: Path,
    pipeline_prefix: str,      # "cat12" or "qsirecon-{workflow}"
    modality: str,             # "anat" or "dwi"
    extra_entities: list[str], # interface-specific BIDS entities
) -> Path:
```

### 3. `_write_output()` - Identical logic

**Files:**
- `interfaces/cat12/cat12.py:153-166`
- `interfaces/qsirecon/qsirecon.py:157-170`
- `interfaces/cat12/cli.py:256-266` (third copy!)

All three: call `_build_output_path()`, create parent dirs, write TSV, log.

**Action:** Extract to `interfaces/utils.py`. This becomes trivial once `build_output_path()` is shared.

### 4. `_run_recon()` - Near-identical workflow

**Files:**
- `interfaces/cat12/cat12.py:169-200`
- `interfaces/qsirecon/qsirecon.py:173-218`
- `interfaces/cat12/cli.py:269-334` (variant)

All follow the same pattern: iterate plan, check for existing outputs, build pending plan, run workflow, write outputs. QSIRecon variant adds a try/except around `_build_output_path`.

**Action:** Extract shared logic to `interfaces/utils.py`:

```python
def run_recon_with_caching(
    recon: ReconInput,
    plan: dict[AtlasDefinition, list],
    config: ParcellationConfig,
    build_output_path_fn: Callable,
    write_output_fn: Callable,
    run_workflow_fn: Callable,
) -> list[Path]:
```

### 5. `run_parcellations()` - Similar orchestration

**Files:**
- `interfaces/cat12/cat12.py:203-229`
- `interfaces/qsirecon/qsirecon.py:221-264`

Both: configure logging, call loader, check empty, optionally parallelize with `ProcessPoolExecutor`, flatten outputs.

**Action:** Extract shared orchestration loop:

```python
def run_parcellations_generic(
    config: ParcellationConfig,
    load_fn: Callable,
    run_single_fn: Callable,
) -> list[Path]:
```

### 6. `_parse_log_level()` - Duplicate definition

**Files:**
- `interfaces/utils.py:12-40` (shared version)
- `interfaces/cat12/cli.py:173-177` (local copy)

**Action:** Remove the local copy in `cli.py`, import from `interfaces/utils`.

### 7. `load_config()` - Similar TOML parsing

**Files:**
- `interfaces/cat12/cat12.py:38-83`
- `interfaces/qsirecon/qsirecon.py:39-83`

Both parse TOML with the same fields. Key differences:
- CAT12's `load_config` takes `Path`, QSIRecon's takes `argparse.Namespace`
- QSIRecon supports CLI argument overrides

**Action:** Extract shared TOML parsing core, let each interface handle its own CLI overlay.

---

## Estimated Impact

| Metric | Before | After |
|--------|--------|-------|
| Duplicated LOC | ~250 | ~30 |
| Files with duplication | 5 | 0 |
| Shared utility functions | 2 | ~8 |

## Risks

- Need to ensure `ScalarMapDefinition` differences (CAT12 has `tissue_type`, QSIRecon has `param`/`model`/`desc`/`recon_workflow`) are handled cleanly in shared code
- The `cat12/cli.py` has its own parallel processing path (CSV-based) that duplicates the TOML-based path — consider whether these should converge

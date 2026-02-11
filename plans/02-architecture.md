# Plan 02: Architecture Improvements

## 2.1 Config Dataclass Consolidation

### Problem

`Cat12Config` and `QSIReconConfig` are nearly identical dataclasses (11/12 fields match). They share the same field names, types, and defaults, but are defined separately.

**Files:**
- `interfaces/cat12/models.py:41-57` (`Cat12Config`)
- `interfaces/qsirecon/models.py:31-47` (`QSIReconConfig`)

### Action

Create a shared base config in `interfaces/models.py`:

```python
@dataclass
class ParcellationConfig:
    input_root: Path
    output_dir: Path
    atlases: list[AtlasDefinition] | None = None
    subjects: list[str] | None = None
    sessions: list[str] | None = None
    mask: Path | str | None = None
    background_label: int = 0
    resampling_target: str | None = "data"
    force: bool = False
    log_level: int = logging.INFO
    n_jobs: int = 1
    n_procs: int = 1
```

Then have `Cat12Config(ParcellationConfig)` add only `mask: Path | str | None = "gm"` as an override, and `QSIReconConfig` can simply be an alias or empty subclass.

**Benefit:** This also replaces the ad-hoc `ParcellationConfig` Protocol in `interfaces/runner.py:49-59` with a concrete base class.

---

## 2.2 ScalarMapDefinition Interface

### Problem

The shared `interfaces/models.py` uses a TYPE_CHECKING hack for `ScalarMapDefinition`:

```python
if TYPE_CHECKING:
    ScalarMapDefinition = Any
else:
    ScalarMapDefinition = "ScalarMapDefinition"
```

This breaks type safety. `ReconInput.scalar_maps` is typed as `Sequence[ScalarMapDefinition]`, but the actual type depends on which interface you're using (CAT12 or QSIRecon).

### Action

Define a `ScalarMapProtocol` (or a shared base dataclass) in `interfaces/models.py` with the common fields:

```python
@dataclass(frozen=True)
class ScalarMapBase:
    name: str
    nifti_path: Path
    space: str | None = None
    desc: str | None = None
```

Then have interface-specific classes inherit from it:
- `cat12/models.py`: `class ScalarMapDefinition(ScalarMapBase)` adds `tissue_type`
- `qsirecon/models.py`: `class ScalarMapDefinition(ScalarMapBase)` adds `param`, `model`, `origin`, `recon_workflow`

This makes `planner.py` and `runner.py` type-safe by operating on `ScalarMapBase`.

---

## 2.3 Cat12 CLI Consolidation

### Problem

`interfaces/cat12/cli.py` (691 lines) is the largest file in the project and largely duplicates the TOML-based workflow in `interfaces/cat12/cat12.py`. It provides an alternative CSV-based entry point with env-var configuration.

Key duplication:
- `_build_output_path()` — third copy (lines 227-253)
- `_write_output()` — third copy (lines 256-266)
- `process_single_subject()` — reimplements `_run_recon()` (lines 269-334)
- `_parse_log_level()` — reimplements shared util (lines 173-177)
- `config_from_env()` — separate config loading path (lines 180-224)
- Config override logic in `main()` — 60 lines of dataclass reconstruction (lines 586-646)

### Action

1. **Eliminate duplicated helpers** by importing from shared utils (after Plan 01)
2. **Replace config reconstruction** with a mutable config or builder pattern:

```python
# Instead of 5 Cat12Config(...) reconstructions:
config = config_from_env()
if args.root:
    config.input_root = args.root.expanduser().resolve()
if args.output_dir:
    config.output_dir = args.output_dir.expanduser().resolve()
if args.force:
    config.force = True
```

This requires making `Cat12Config` a non-frozen dataclass (it already is).

3. **Reuse `_run_recon()`** from `cat12.py` instead of reimplementing in `process_single_subject()`

**Estimated reduction:** ~200 lines from `cli.py`

---

## 2.4 Planner/Runner Re-export Pattern

### Problem

`cat12/planner.py` and `cat12/runner.py` are 6-line files that only re-export from `interfaces/planner.py` and `interfaces/runner.py` under new names:

```python
# cat12/planner.py
from parcellate.interfaces.planner import plan_parcellation_workflow as plan_cat12_parcellation_workflow
```

Similarly for `qsirecon/planner.py` and `qsirecon/runner.py`.

### Action

Two options:

**Option A (simpler):** Delete the re-export files. Import directly from `interfaces.planner` / `interfaces.runner` where needed:
```python
from parcellate.interfaces.planner import plan_parcellation_workflow
from parcellate.interfaces.runner import run_parcellation_workflow
```

**Option B (if interface-specific overrides are planned):** Keep the files but document that they exist for future extensibility.

**Recommendation:** Option A — remove 4 files, simplify imports.

---

## 2.5 Entry Point Consistency

### Problem

The two interfaces have different CLI patterns:
- CAT12 TOML CLI: `parcellate cat12 config.toml` (via `cli.py:main`)
- CAT12 CSV CLI: `parcellate-cat12 subjects.csv` (via `cat12/cli.py:main`)
- QSIRecon CLI: `parcellate qsirecon --config config.toml --input-root /path` (via `qsirecon.py:main`)

QSIRecon supports both TOML config + CLI arg overrides. CAT12 TOML mode only accepts a config file. The CAT12 CSV CLI is a completely separate path.

### Action

1. Add CLI argument overrides to the CAT12 TOML path (matching QSIRecon's pattern)
2. Consider merging the CSV mode into the main CLI as `parcellate cat12 --csv subjects.csv`
3. Register a `parcellate-qsirecon` entry point for parity with `parcellate-cat12`

---

## 2.6 Unused Dependency: `seaborn`

### Problem

`seaborn>=0.13.2` is listed as a dependency in `pyproject.toml:25` but is never imported in any source file.

### Action

Remove from `pyproject.toml` dependencies list.

---

## 2.7 Unused Dependency: `scipy` (Implicit)

### Problem

`metrics/volume.py` imports `scipy.stats` (line 7), but `scipy` is not listed as an explicit dependency in `pyproject.toml`. It's currently pulled in transitively through `nilearn`, which depends on `scipy`.

### Action

Add `scipy` as an explicit dependency in `pyproject.toml` to prevent breakage if nilearn ever drops it.

---

## 2.8 Module `__init__.py` Files

### Problem

Most `__init__.py` files are empty or minimal. Key public APIs are not re-exported.

**Current state:**
- `parcellate/__init__.py` — not checked
- `interfaces/__init__.py` — empty
- `metrics/__init__.py` — likely empty
- `parcellation/__init__.py` — likely empty
- `utils/__init__.py` — exports `_load_nifti`

### Action

Add re-exports to make imports cleaner:

```python
# parcellate/__init__.py
from parcellate.parcellation.volume import VolumetricParcellator

# parcellate/metrics/__init__.py
from parcellate.metrics.volume import BUILTIN_STATISTICS
from parcellate.metrics.base import Statistic
```

This allows `from parcellate import VolumetricParcellator` instead of the deep import.

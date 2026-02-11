# Plan 05: Testing Improvements

## Current Coverage

| Test File | Approx Tests | Coverage |
|-----------|-------------|----------|
| `test_metrics.py` | 72 | Thorough: all 13 builtin stats, edge cases |
| `test_parcellator.py` | 38 | Good: fit/transform, masking, resampling, caching |
| `test_interfaces_cat12.py` | ~13 | Loader and model tests |
| `test_interfaces_qsirecon.py` | ~13 | Loader and model tests |

## Gaps and Additions

### 5.1 CLI Tests (Missing)

**Priority: HIGH**

No tests exist for any CLI entry point:
- `cli.py:main()` — the unified CLI dispatcher
- `cat12/cat12.py:main()` — CAT12 TOML CLI
- `cat12/cli.py:main()` — CAT12 CSV CLI
- `qsirecon/qsirecon.py:main()` — QSIRecon CLI

**Tests to add:**

```
tests/test_cli.py:
  - test_main_no_args_prints_help
  - test_main_cat12_subcommand_delegates
  - test_main_qsirecon_subcommand_delegates
  - test_main_unknown_command_returns_1

tests/test_cat12_cli.py:
  - test_load_config_from_toml
  - test_load_config_missing_file
  - test_run_parcellations_empty_inputs
  - test_run_parcellations_with_force

tests/test_qsirecon_cli.py:
  - test_load_config_from_args
  - test_load_config_from_toml
  - test_load_config_cli_overrides_toml
  - test_run_parcellations_empty_inputs

tests/test_cat12_csv_cli.py:
  - test_load_subjects_from_csv_valid
  - test_load_subjects_from_csv_missing_column
  - test_sanitize_subject_code
  - test_sanitize_session_id_nan
  - test_config_from_env
  - test_config_from_env_missing_root
  - test_parse_atlases_from_env
  - test_dry_run_mode
  - test_main_with_csv_and_env
```

### 5.2 Integration Tests (Missing)

**Priority: MEDIUM**

No end-to-end tests that exercise the full pipeline from config loading through output writing. Current tests mock individual components.

**Tests to add:**

```
tests/test_integration.py:
  - test_cat12_full_pipeline_synthetic_data
    - Create temp directory with synthetic NIfTI files
    - Write TOML config
    - Call run_parcellations()
    - Verify TSV outputs exist and contain expected columns

  - test_qsirecon_full_pipeline_synthetic_data
    - Create temp directory mimicking QSIRecon layout
    - Write TOML config with atlas definitions
    - Call run_parcellations()
    - Verify outputs

  - test_force_flag_overwrites_existing
  - test_skip_existing_outputs
```

### 5.3 Planner Tests (Sparse)

**Priority: LOW**

`interfaces/planner.py` has no dedicated test file.

**Tests to add:**

```
tests/test_planner.py:
  - test_plan_matches_by_space
  - test_plan_no_match_different_spaces
  - test_plan_empty_atlases
  - test_plan_empty_scalar_maps
  - test_space_match_case_insensitive
  - test_space_match_none_values
```

### 5.4 Runner Tests (Sparse)

**Priority: LOW**

`interfaces/runner.py` has no dedicated test file.

**Tests to add:**

```
tests/test_runner.py:
  - test_validate_scalar_map_spaces_consistent
  - test_validate_scalar_map_spaces_inconsistent_raises
  - test_validate_scalar_map_spaces_empty
  - test_run_workflow_single_atlas
  - test_run_workflow_parcellator_init_failure_skips
  - test_run_workflow_parallel_execution
  - test_parcellate_scalar_map_exception_returns_none
```

### 5.5 Skewness/Kurtosis Metric Tests

**Priority: LOW**

Once wired into BUILTIN_STATISTICS (see Plan 03, section 3.3), add tests:

```
tests/test_metrics.py (additions):
  - test_skewness_symmetric_distribution
  - test_skewness_positive_skew
  - test_skewness_nan_handling
  - test_kurtosis_normal_distribution
  - test_kurtosis_heavy_tails
  - test_kurtosis_nan_handling
```

### 5.6 Test Fixture Improvements

**Current:** `conftest.py` creates synthetic NIfTI images and dataframes. This is good.

**Improvements:**
- Add a fixture that creates a full synthetic BIDS-derivative directory structure for integration tests
- Add parametric fixtures for different atlas sizes (small: 5 regions, medium: 100, large: 1000)
- Add a fixture for synthetic TOML config files

### 5.7 Performance/Regression Tests (Optional)

**Priority: LOW**

Add timing benchmarks for key operations:
- `VolumetricParcellator.fit()` + `transform()` with various atlas sizes
- Full pipeline with 10 subjects
- Use `pytest-benchmark` or simple timing assertions

---

## Summary

| Area | Current Tests | Tests to Add | Priority |
|------|--------------|-------------|----------|
| CLI entry points | 0 | ~20 | High |
| Integration (end-to-end) | 0 | ~5 | Medium |
| Planner | 0 | ~6 | Low |
| Runner | 0 | ~7 | Low |
| Skewness/Kurtosis | 0 | ~6 | Low |
| **Total** | **~136** | **~44** | |

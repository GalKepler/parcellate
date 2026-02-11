# Parcellate Refactoring Plan

## Summary

This plan outlines a comprehensive refactoring of the `parcellate` project across six areas:

| Plan | Focus | Priority | Effort |
|------|-------|----------|--------|
| [01-code-deduplication.md](01-code-deduplication.md) | Eliminate duplicated code across CAT12/QSIRecon interfaces | High | Medium |
| [02-architecture.md](02-architecture.md) | Improve module design, config handling, and extensibility | Medium | Large |
| [03-bugs-and-correctness.md](03-bugs-and-correctness.md) | Fix bugs and semantic issues in current code | High | Small |
| [04-performance.md](04-performance.md) | Optimize hot paths and resource usage | Medium | Small |
| [05-testing.md](05-testing.md) | Expand test coverage and add integration tests | Medium | Medium |
| [06-documentation.md](06-documentation.md) | Improve docstrings, README, and user guides | Low | Medium |

## Current State

- **28 source files**, ~3,200 LOC
- **5 test files**, ~2,200 LOC, 136 tests
- **Architecture:** Two parallel interface pipelines (CAT12, QSIRecon) sharing a core parcellation engine
- **Active WIP:** Uncommitted changes in `metrics/volume.py`, `qsirecon/loader.py`, `qsirecon/qsirecon.py`
- **Recent work:** pybids removal, parallelization, shared planner/runner extraction

## Recommended Order of Execution

1. **Phase 1 - Bugs & correctness** (plan 03) - Fix before anything else
2. **Phase 2 - Code deduplication** (plan 01) - Reduces maintenance surface
3. **Phase 3 - Architecture** (plan 02) - Build on the deduplicated base
4. **Phase 4 - Performance** (plan 04) - Optimize once the structure is clean
5. **Phase 5 - Testing** (plan 05) - Lock in correctness after refactoring
6. **Phase 6 - Documentation** (plan 06) - Document the final state

# scPolish violinplot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-mutating `scp.violinplot()` for standard and stacked grouped distributions, with the publication styling and PBMC3k examples specified in the design.

**Architecture:** Read expression and numeric metadata through `scanpy.get.obs_df()`, then use one Seaborn/Matplotlib rendering path for standard and stacked modes. Reuse scPolish's `STYLE` and `export()`, and centralize category-color lookup so `dimplot()` and violin plots share colors.

**Tech Stack:** Python 3.11+, AnnData, Scanpy, Seaborn, Matplotlib, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-11-06-violinplot-design.md`

## Global Constraints

- Use `scanpy.get.obs_df()` for `adata.X`, layers, `.raw`, and numeric `obs` fields; do not normalize or scale data.
- Do not modify input AnnData or global Matplotlib settings.
- Figure size follows the feature grid unless `figsize` is supplied; export DPI is `300`, and points are off by default.
- Keep vector violin bodies and text; rasterize only individual points when requested.
- Use Glasbey for adaptive categorical palettes; do not add unrelated dependencies, statistical testing, custom density estimation, or a new theme system.
- Keep generated files under `examples/output/violinplot/` out of version control.
- Preserve all pre-existing working-tree changes.

## Review Focus

- Sparse `X` and layer/raw sources must produce numeric values without densifying the full matrix; pin this in the source-handling tests.
- Missing grouping and split values must be shown in gray, and missing feature values omitted; none may be written back. Pin this in the immutability/missingness tests.
- Ordered categoricals with unused levels must retain their declared order; pin this in the ordering test.
- Explicit group order and custom palette must stay aligned when categories are reordered; pin this in the palette/order test.
- An incompatible `layer` plus `use_raw=True` must fail clearly; pin this in the source-conflict test.

---

### Task 1: Plot API contract tests

**Files:**
- Create: `tests/test_violinplot.py`
- Modify: `tests/conftest.py` only if the existing AnnData fixture needs a numeric layer or raw setup

**Interfaces:**
- Consumes: existing `adata` fixture and public `scp`/`scpolish.pl` imports.
- Produces: tests defining `violinplot(adata, features, groupby, *, mode, split_by, layer, use_raw, palette, order, inner, points, orientation, ncols, figsize, ax, save, dpi)` and its Axes/figure return values.

- [x] **Step 1: Write failing API tests** for root and `scpolish.pl` exports, one/multiple features, standard/stacked axes shapes, custom category order, split categories, horizontal orientation, numeric metadata, missing values, dense and sparse `X`, layer/raw source selection, PDF export, point rasterization, and AnnData/rcParams immutability.
- [x] **Step 2: Run focused tests to verify the new API is absent or fails**

Run: `uv run --no-sync pytest tests/test_violinplot.py -q`

Expected: failures identify missing exports/function and unimplemented behaviors.

### Task 2: Shared categorical palette resolution

**Files:**
- Modify: `src/scpolish/_utils/colors.py`
- Modify: `src/scpolish/pl/_dimplot.py`
- Test: `tests/test_core_plots.py`

**Interfaces:**
- Consumes: `AnnData`, category order, and optional palette (`str`, sequence, or mapping).
- Produces: a shared resolver that returns one deterministic color per category, defaulting to exact-size cached muted Glasbey palettes and supporting pastel, Scanpy, and validated custom palettes.

- [x] **Step 1: Run GitNexus impact before changing `dimplot()` or `categorical_colors()`**; because the current index is behind and `categorical_colors` impact may be `UNKNOWN`, confirm callers with `rg` before editing.
- [x] **Step 2: Add or update palette tests** proving explicit, stored, default, and mapping palettes resolve in category order and remain consistent between `dimplot()` and `violinplot()`.
- [x] **Step 3: Implement the shared resolver and route `dimplot()` through it**, preserving current outputs and never writing generated colors into the source AnnData.
- [x] **Step 4: Run the existing plotting and palette tests**

Run: `uv run --no-sync pytest tests/test_core_plots.py -q`

Expected: PASS with existing `dimplot()` colors and the new resolver behavior aligned. The new API tests remain red until Task 3 supplies `violinplot()`.

### Task 3: Standard and stacked violin implementation

**Files:**
- Create: `src/scpolish/pl/_violinplot.py`
- Modify: `src/scpolish/pl/__init__.py`
- Modify: `src/scpolish/__init__.py`
- Test: `tests/test_violinplot.py`

**Interfaces:**
- Consumes: shared palette resolver, `STYLE`, `export()`, `AnnData`, and `scanpy.get.obs_df()`.
- Produces: `violinplot(...) -> Axes | tuple[Figure, np.ndarray]`; a single feature returns one `Axes`, multiple features return a 2D axes array with hidden unused panels where applicable.

- [x] **Step 1: Add failing edge tests** for empty/duplicate features, unknown grouping or split fields, invalid modes/orientations/orders, missing split values, and external axes compatibility.
- [x] **Step 2: Implement value extraction and local tidy data** using `use_raw = adata.raw is not None and layer is None` when `use_raw` is omitted; reject `layer` with `use_raw=True`; do not copy the full expression matrix.
- [x] **Step 3: Implement standard grids** with `ncols`, group palette, vertical/horizontal axes, split halves for exactly two split levels and dodged violins for more levels, optional inner boxes and points, automatic category-label rotation, and vector/raster artist handling.
- [x] **Step 4: Implement stacked mode** with one shared-category axes row per feature, independent feature value scales, shared order, and category tick labels shown once.
- [x] **Step 5: Apply scoped `STYLE`, thin outlines, white background, minimal spines, no grid, `export()`, and return axes without closing figures.**
- [x] **Step 6: Run the focused suite**

Run: `uv run --no-sync pytest tests/test_violinplot.py tests/test_core_plots.py -q`

Expected: PASS, including all source, order, split, missingness, export, and immutability assertions.

### Task 4: Public exports, README, and PBMC3k example

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Create: `examples/violinplot.py`
- Generate but do not track: `examples/output/violinplot/*.pdf` and `*.png`

**Interfaces:**
- Consumes: the completed `scp.violinplot()` API.
- Produces: public root and plotting-module exports, concise README usage, a runnable PBMC3k script, and five figure variants saved as PDF and PNG.

- [x] **Step 1: Export and document the API**, including standard and stacked examples, source behavior, split behavior, return types, and axes customization.
- [x] **Step 2: Add a PBMC3k script** that inspects available variables, selects existing genes and grouping metadata, creates a local named synthetic split only if needed, and saves standard, 2x2 multi-feature, stacked, horizontal, and split figures as both PDF and PNG at 300 DPI.
- [x] **Step 3: Ignore only `examples/output/violinplot/`** so rendered deliverables remain on disk but outside version control.
- [x] **Step 4: Run all requested checks**

Run: `uv run --no-sync pytest -q && uv run --no-sync ruff check .`

Expected: all tests pass and Ruff reports no issues.

- [x] **Step 5: Render and inspect every PBMC3k PDF/PNG** for palette consistency, category order, split labels, readability, spacing, and clipping; adjust layout values and rerender if needed.
- [x] **Step 6: Re-run focused and full tests after any visual fixes**, then report generated file paths, inspection results, test/lint results, and limitations.

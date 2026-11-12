# Violin Style Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Match the supplied ggpubr/SCpubr violin references with filled violins, dark outlines, a narrow white box overlay, readable optional points and labels, and optional supplied-p-value brackets.

**Architecture:** Keep the current Seaborn/Matplotlib renderer. Refine the existing layers, add optional `statannotations` support for caller-supplied p-values, add regression tests, then refresh the existing publication example and README.

**Tech Stack:** Python, Seaborn >=0.13, Matplotlib, adjustText, pytest.

**Spec:** `docs/superpowers/specs/2026-11-12-violin-style-design.md`

## Global Constraints

- Reuse the existing Seaborn and Matplotlib renderer, muted/pastel palette, scoped style context, layout and export helpers.
- Keep `inner=None`, `cut=0`, and the white box overlay at width `0.22`; optional brackets use supplied values only.
- Keep points opt-in, dark grey, and area 18. Preserve labels at fontsize 6 with `adjustText`, paired lines, and observation counts.
- Preserve AnnData, avoid global style changes, never calculate statistical tests automatically, and do not aggregate cells into donors.
- No new plotting framework, style preset API, or automatic point display.

## Review Focus

- Two-level split violins with boxes: each hue must retain a distinct aligned summary box.
- Labelled observations near points and white boxes: repulsion must reduce overlap without changing labels or values.
- Sparse or missing category observations: keep existing category, missing-value, and pairing behavior.
- Matplotlib settings and AnnData state: rendering must remain scoped and non-mutating.
- Dense point collections: preserve optional points and the existing rasterization threshold.

---

### Task 1: Refine violin layers and pin their behavior

**Files:**
- Modify: `src/scpolish/pl/_violinplot.py`
- Test: `tests/test_violinplot.py`

**Interfaces:**
- Consumes: existing `violinplot()` arguments and the existing Seaborn/adjustText dependencies.
- Produces: same return types and arguments; dark violin outlines, white boxes aligned to split hues, and point-aware label adjustment.

- [x] **Step 1: Add regression tests** for dark violin edges, the 0.22 white box width, distinct box centers for two split hues, and passing observation coordinates to label adjustment; retain checks that points default off and have area 18 when enabled.
- [x] **Step 2: Run the focused tests** with `uv run pytest tests/test_violinplot.py -q`; the new style, split-alignment, and label-repulsion cases failed against the original renderer.
- [x] **Step 3: Refine the Seaborn layers** in `_violinplot.py`: set near-black violin outlines with restrained line width, separate split-hue box overlays, retain dark-grey area-18 optional points, and pass plotted observation coordinates into `adjust_text` while preserving label targets.
- [x] **Step 4: Run the focused tests** with `uv run pytest tests/test_violinplot.py -q`; all 30 passed after the change.

### Task 2: Refresh the publication example and documentation

**Files:**
- Modify: `examples/publication_defaults.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: the existing `violinplot()` API, plus opt-in `pairs` and caller-supplied `pvalues`.
- Produces: examples and documentation showing the chosen style and optional annotations without running tests.

- [x] **Step 1: Add optional annotation support** with explicit `pairs` and `pvalues`; test supplied values and invalid inputs while keeping statistical tests opt-in nowhere in scPolish.
- [x] **Step 2: Confirm the donor example** shows dark observation points, labels, paired lines, and counts; retain synthetic observation-level data without aggregation.
- [x] **Step 3: Update the README** to describe the dark violin outline, white box overlay, and optional supplied-p-value annotations.
- [x] **Step 4: Render the publication gallery** with `examples/publication_defaults.py`; confirm PDF/PNG outputs and no AnnData mutation assertion failure.
- [x] **Step 5: Visually inspect** categorical UMAPs, score overlays, standard/donor violins, and the six-feature grid for colors, overlap, and clipping.
- [ ] **Step 6: Run full verification** with `pytest -q`, `ruff check .`, `ruff format --check .`, and `git diff --check`; require all checks to pass.
- [ ] **Step 7: Run Cavecrew review and graph impact analysis** with GitNexus `detect_changes --scope all`; review affected flows, then commit all authorized plotting changes and this plan on `main`. Do not push.

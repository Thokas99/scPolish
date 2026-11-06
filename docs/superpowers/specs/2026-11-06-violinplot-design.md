# scPolish violinplot design

## Intent

Add `scp.violinplot()` as the next Scanpy/AnnData plotting function in scPolish. It should make common grouped expression and metadata distributions ready to export while returning ordinary Matplotlib axes for follow-up edits.

## Goals

- Plot one or more genes or numeric `adata.obs` fields in standard or vertically stacked layouts.
- Support categorical grouping, explicit category ordering, custom palettes, split categories, horizontal orientation, optional box interiors, and optional cell points.
- Read `adata.X`, a named layer, or `.raw` using Scanpy's expression lookup conventions, without normalizing or scaling values.
- Leave the input AnnData and global Matplotlib settings unchanged.
- Match scPolish's typography, figure export, and group colors; keep the output suitable for vector PDF export, with point clouds rasterized only when enabled.
- Include focused tests, concise README examples, and PBMC3k-generated PDF/PNG examples that are visually inspected and excluded from version control.

## Non-goals and anti-goals

- No statistical tests, significance annotations, density-estimation implementation, normalization, or new theme system.
- No additional plotting functions or dependency additions.
- Do not retain generated PBMC figures in version control.
- Do not claim PBMC visual validation if the dataset cannot be obtained or the figures cannot be rendered and inspected.

## Constraints and current architecture

- The package exports plotting functions from `src/scpolish/pl/__init__.py` and the package root `src/scpolish/__init__.py`.
- `dimplot()` and `featureplot()` share `STYLE`, `export()`, and small axes helpers in `src/scpolish/pl/_styles.py`.
- `featureplot()` uses `scanpy.get.obs_df()` to resolve genes, numerical metadata, layers, and `.raw`, then copies plot values into a local object. `dimplot()` uses Scanpy's palette resolution and the current `categorical_colors()` helper is not used by the plotting functions.
- The embedding panel helper requires coordinates and is not appropriate for violin layouts.
- Seaborn is already a project dependency. The project virtualenv has Scanpy 1.12.5; global Python does not have Scanpy.
- The current working tree contains unrelated uncommitted changes. Implementation must preserve and build on them.

## Approaches

1. **Scanpy wrappers with fallbacks.** Use `sc.pl.violin()` and `sc.pl.stacked_violin()` for their native cases, then fall back to Seaborn for split/horizontal layouts and numeric metadata in stacked mode. This uses Scanpy plotting APIs more directly but creates multiple behavior paths and makes shared palette/layout guarantees harder.
2. **Shared Seaborn plotting path (selected).** Use `scanpy.get.obs_df()` for AnnData source resolution and Seaborn/Matplotlib for standard and stacked layouts. This keeps one layout and color path for genes and numeric metadata, and supports split/horizontal modes and supplied axes consistently. It gives up Scanpy's plot wrappers where they cannot express the requested styling/layout together.

The user selected approach 2. Use Scanpy's `sc.pl.violin()` and `sc.pl.stacked_violin()` only if implementation proves a requested feature can be delegated without creating a divergent path; otherwise keep both modes on the shared Seaborn path.

## Behavior and API

The public call follows the requested API and adds `ax=None` for a compatible externally supplied single-panel axes:

```python
scp.violinplot(
    adata,
    features,
    groupby,
    *,
    mode="standard",
    split_by=None,
    layer=None,
    use_raw=None,
    palette="muted",
    order=None,
    inner="box",
    points=False,
    orientation="vertical",
    ncols=2,
    figsize=None,
    ax=None,
    save=None,
    dpi=300,
)
```

- Accept a string or a non-empty sequence of unique feature names. Resolve all requested values with `scanpy.get.obs_df()`; use `adata.raw` by default only when raw exists and no layer is requested, matching `featureplot()`.
- Do not copy the full expression matrix or write categories, palettes, or values to the input. Use a local tidy DataFrame for rendering. Missing feature values are omitted from density calculations; missing group and split values are labeled `NA` and colored gray.
- Preserve ordered categorical levels by default. For non-categorical grouping fields, use deterministic sorted categories, matching `dimplot()`. `order` overrides that order and must contain known categories.
- Resolve category colors through the shared palette resolver as `dimplot()`: muted Glasbey by default, optional pastel Glasbey, validated category dictionaries or ordered color lists, and explicit Scanpy-compatible colors. Standard and stacked plots use the same group colors. A split plot uses the complete split-category set as its hue palette.
- In standard mode, draw one feature per panel in a `ncols` grid. One panel returns an `Axes`; multiple panels return `(fig, axes)` with a two-dimensional axes array. `ax` is accepted only for one feature.
- `split_by` draws split groups side by side; two split categories may be rendered as split halves, while more than two are dodged. The split levels are labeled. Horizontal orientation swaps the category and value axes.
- In stacked mode, draw one shared-category panel per feature in a vertical stack, share the category ordering, and show repeated category labels only once. Keep feature value scales independent because values are not transformed. `ncols` does not affect stacked mode.
- `inner` is passed to Seaborn for the optional box/median/IQR interior. Individual cell points are absent by default; when enabled, rasterize only their collection. Keep violin bodies, axes, labels, and box overlays vector-based.
- Apply `STYLE` in a scoped Matplotlib context, thin outlines, white axes, no grid, minimal spines, balanced layout, and automatic category-label rotation for crowded labels. Save through `export()` with the requested DPI and the existing extension-based format behavior.
- Return the single `Axes` for one feature and `(fig, axes)` for multiple features so callers can customize the result.

## Files and checks in scope

- Add `src/scpolish/pl/_violinplot.py` and export `violinplot` from both package entry points.
- If needed to guarantee color agreement, update the existing palette resolver and route `dimplot()` through the same resolver. Before editing any existing function, run GitNexus impact and confirm its callers; the current index is behind and any `UNKNOWN` result must be checked with source search.
- Add focused tests for exports, one/multiple features, standard/stacked modes, category order, split behavior, layers/raw, numeric metadata, missing values, PDF export, rasterization, axes, and input/global-state immutability.
- Update `README.md` with short standard and stacked examples.
- Add a reproducible PBMC3k example script. Inspect available genes and metadata before choosing features; produce standard, multi-feature, stacked, horizontal, and split examples as PDF and PNG at 300 DPI in `examples/output/violinplot/`. Add only that generated output path to `.gitignore`.
- Run the requested test suite and lint check. Render and inspect the PBMC figures, then correct visible clipping, spacing, label, order, legend, or palette problems.

## Calls left to the builder

- Choose the smallest compatible way to lay out stacked Seaborn axes and remove redundant category labels.
- Choose label rotation threshold and precise spacing values based on readability in the rendered PBMC figures.
- Use PBMC3k metadata actually available in the dataset; add a clearly named synthetic split column only if no suitable split field exists, and keep it local to the example object.
- Keep generated PDFs and PNGs ignored; report their paths and inspection results without adding them to the repository diff.

# scPolish

*Publication-minded single-cell plots with compact, editable defaults.*

scPolish adds consistent plotting helpers to the Scanpy and AnnData ecosystem.
It returns ordinary Matplotlib axes and figures, so plots remain customizable
with familiar Python tools.

The API includes Scanpy-backed embeddings and feature plots, plus grouped
violin plots built with Matplotlib and Seaborn.

## What it values

- **Scanpy and AnnData native:** use your AnnData object and existing coordinates, genes, and metadata.
- **Minimal code:** useful defaults with options for common figure adjustments.
- **Publication-minded:** compact layouts and vector points by default, with optional rasterization for smaller files.
- **Composable:** get a Matplotlib `Axes` back and keep working with the usual tools.
- **Non-destructive:** plotting does not retain Scanpy's generated categorical palette in `adata.uns`.
- **Reproducible:** use the embedding and data you already have; scPolish does not silently downsample cells.

## Installation

### From a checkout

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or install the local project with pip:

```bash
python -m pip install -e .
```

scPolish is not published on PyPI yet. When it is, this section will get a proper package-install command. Until then, the checkout is the package.

## Quick Start

```python
import scanpy as sc
import scpolish as scp

adata = sc.datasets.pbmc3k_processed()
scp.dimplot(adata, groupby="louvain", save="dimplot.pdf")
scp.featureplot(
    adata,
    features=["MS4A1", "NKG7", "FCGR3A"],
    ncols=2,
    save="featureplot.pdf",
)
```

The plotting functions are also available under `scpolish.pl`. Embedding plots
use existing coordinates; `basis="umap"`, `"pca"`, and `"tsne"` select an
embedding without computing it.
The original `scp.embedding()` API remains available.

```python
import matplotlib.pyplot as plt

scp.dimplot(adata, "louvain", labels="direct")
scp.dimplot(adata, "louvain", labels="numbered")
scp.dimplot(adata, "louvain", raster=True)  # opt into rasterized points
# For an object with sample annotations:
# scp.dimplot(adata, "cell_type", split_by="sample_id")
scp.featureplot(adata, ["MS4A1", "NKG7", "FCGR3A"], ncols=2)
fig, ax = plt.subplots(figsize=(6, 5))
scp.dimplot(adata, "louvain", ax=ax)
ax.set_title("My atlas")
fig.savefig("custom.pdf", dpi=300)
```

See [the runnable PBMC3k examples](examples/core_plots.py) for categorical UMAPs,
direct and numbered labels, a clearly marked synthetic split, single and multiple
gene feature plots, and existing Matplotlib axes. The script saves PDF and PNG
figures at 300 DPI under `examples/output/core_plots/`.
The lower-level `scp.embedding()` examples in
[basic_embedding.py](examples/basic_embedding.py) also use PBMC3k and save
categorical, numeric-metadata, and gene-expression UMAPs under
`examples/output/basic_embedding/`.

Single panels return `Axes`; multiple panels return `(fig, axes)`, where `axes`
is always a two-dimensional NumPy array, including hidden unused panels.
A supplied `ax` requires a single panel. Figures remain open; nothing calls
`plt.show()`. Save through `save=` for scoped PDF typography and 300 DPI export.
Extensionless save paths use PDF. PDF and PNG exports retain the requested
figure dimensions.

Embedding panels default to 4.5 × 5.5 inches to fit the usual tall UMAP shape.
Multi-panel layouts grow with their rows and columns; pass `figsize` to set the
exact total canvas size. Default categorical legends reserve room beside the
embedding. Default layouts preserve embedding proportions; explicit `figsize`
or supplied axes fills the panel and may stretch coordinates to match its shape.
Multi-feature grids use independent scales; use
`shared_scale=True` only for comparable quantities. `layer` and `use_raw` follow
Scanpy expression-source rules. Numeric `obs` columns are also supported.
Feature plots use the magma colormap and compact colorbars by default; pass
`cmap=` to choose another map. Values at or below `threshold=0` are gray; pass
`threshold=None` to color negative scores. Missing values also stay gray.

## Categorical palettes

`dimplot()` and `violinplot()` share dynamically generated Glasbey colors. The
default is `palette="muted"`; `palette="pastel"` selects lighter colors.
Automatic palettes generate exactly one distinct color per observed category,
and an annotation keeps the same mapping across both plot types. Observed
categories retain their declared pandas categorical order; other values sort
deterministically. Unused categories are excluded and missing cells are gray.
Palette resolution never changes `adata.uns`.

```python
scp.dimplot(adata, "louvain", palette="pastel")
scp.violinplot(adata, ["CD3D", "MS4A1"], "louvain", palette="muted")

colors = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
]
custom = dict(zip(adata.obs["louvain"].cat.categories, colors, strict=True))
scp.dimplot(adata, "louvain", palette=custom)
scp.violinplot(
    adata,
    ["CD3D", "MS4A1"],
    "louvain",
    palette=colors,
)

scp.dimplot(adata, "louvain", palette="scanpy")  # stored colors or Scanpy defaults
scp.dimplot(adata, "louvain", palette=["#4E79A7"])  # warns and falls back
```

Color lists follow the resolved category order: `violinplot(order=...)` when
provided, otherwise pandas categorical order or deterministic sorting. Dicts
match category names regardless of order. Invalid custom colors, missing or
extra dict keys, duplicate colors and incorrect list lengths warn and fall back
to a complete muted palette. Unknown named presets raise `ValueError`.
The example uses PBMC3k's eight `louvain` categories; supply one list color per
category in the resolved order.

Labels use observed cells nearest category medians; automatic mode chooses
direct labels up to 20 categories and numbered labels above that. Split panels
share colors and limits; they skip label repulsion to keep plotting bounded.
Missing split values get their own panel.

Points and text stay vector by default. Set `raster=True` to rasterize point
collections while keeping labels, legends, axes and colorbar text editable in
vector exports. Point areas shrink as cell counts grow; `point_scale` adjusts
`dimplot()` and `featureplot()` markers, and explicit `size` overrides the
automatic size. Typography uses a temporary rc context. No global theme is
installed.

## Violin plots

`violinplot()` plots genes or numeric `obs` columns by a categorical annotation.
Standard mode makes a feature grid; stacked mode shares the category axis across
features. It reads `X`, layers, or `.raw` without normalizing values or changing
AnnData. Group colors follow `dimplot()`; split categories have their own legend.

```python
scp.violinplot(
    adata,
    ["MS4A1", "CD79A", "S100A8", "NKG7"],
    groupby="louvain",
    ncols=2,
    save="markers.pdf",
)
scp.violinplot(
    adata,
    ["MS4A1", "CD79A", "S100A8", "NKG7", "FCGR3A", "PPBP"],
    groupby="louvain",
    mode="stacked",
    save="markers-stacked.pdf",
)
if "sample_id" in adata.obs:
    scp.violinplot(
        adata,
        "MS4A1",
        groupby="louvain",
        split_by="sample_id",
        orientation="horizontal",
    )
```

Cell points are off by default. `inner="box"` shows median and IQR; set
`points=True` to add cells and `raster=True` to rasterize those points. One
feature returns an `Axes`; multiple features return `(fig, axes)` for
customization. Save as PDF or PNG by choosing the filename extension.

## Roadmap

These are plans, not features hiding behind a very optimistic README.

**v0.1 — Dimensionality reduction**

- Publication-minded embeddings and consistent styling
- Cluster labels and split panels (implemented); population highlighting (planned)
- Continuous feature plots with independent or shared scales (implemented)
- Optional rasterization and split panels (implemented)

**v0.2 — Common single-cell plots**

- Standard and stacked violin plots (implemented); boxplots, dotplots,
  cell-type proportions, and distributions

**v0.3 — Heatmaps**

- Opinionated expression heatmaps
- Marsilea integration and optional PyComplexHeatmap integration

**Future**

- More specialized single-cell visualizations, figure composition, and other publication utilities

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```

## Credits

scPolish builds on the Scanpy, AnnData, Matplotlib, Seaborn, NumPy, and pandas
ecosystems. Their plotting and data tools do the heavy lifting; scPolish helps
them show up to the figure panel looking coordinated.

Documentation guidance: Kassis T, Agarwal V, He Y, Patel D, Brueckner AM.
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
arXiv:2609.00065 (2026). [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065).

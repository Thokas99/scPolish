# scPolish

*Publication-ready single-cell figures with minimal code.*

Single-cell plots often arrive at publication looking like they have just come back from a very long day in the lab. scPolish adds a small layer of thoughtful defaults to Scanpy plots, so you can spend less time nudging Matplotlib and more time deciding what the figure actually says.

It is inspired by packages such as SCpubr and scCustomize, and built for Python's Scanpy and AnnData ecosystem. Scanpy remains in charge of plotting; scPolish tidies the presentation and hands you back a regular Matplotlib axes. No secret rendering engine. No plot captivity. You can still change the title, adjust the axes, or save the result yourself.

## What it values

- **Scanpy-native:** pass an AnnData object and let Scanpy handle coordinates, genes, and metadata.
- **Minimal code:** useful defaults without a settings panel with its own settings panel.
- **Publication-minded:** restrained figure styling and rasterized point clouds by default, with text and axes left editable in vector exports.
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

## Quick start

Your AnnData object needs an existing embedding such as `adata.obsm["X_umap"]`:

```python
import scpolish as scp

ax = scp.embedding(adata, basis="umap", color="cell_type")
ax.figure.savefig("umap.pdf", bbox_inches="tight")
```

The returned axes is ordinary Matplotlib, so feel free to keep polishing:

```python
ax.set_title("My single-cell atlas")
```

### Try it without a dataset

This small example makes its own AnnData object and UMAP-shaped coordinates. The coordinates are synthetic for demonstration; they are not a biological result (and should not be invited to review your manuscript).

```python
import anndata as ad
import numpy as np
import pandas as pd
import scpolish as scp

rng = np.random.default_rng(1)
adata = ad.AnnData(
    rng.poisson(2, (60, 5)),
    obs=pd.DataFrame(
        {
            "cell_type": pd.Categorical(["T", "B"] * 30),
            "score": rng.random(60),
        }
    ),
    var=pd.DataFrame(index=[f"gene{i}" for i in range(5)]),
)
adata.obsm["X_umap"] = rng.normal(size=(60, 2))

ax = scp.embedding(adata, color="cell_type")
ax.figure.savefig("umap.pdf", bbox_inches="tight")
```

## What works today

`scpolish.embedding()` is the first plotting function. It wraps `scanpy.pl.embedding` and supports precomputed Scanpy-compatible embeddings, categorical or continuous observation metadata, and gene-expression coloring. Pass an existing axes or set the new figure size, title, and point size. Scatter collections are rasterized by default to keep dense PDF exports manageable; text and axes remain vector elements. Set `raster=False` if you want vector points too.

The wrapper returns a Matplotlib `Axes`, preserves existing Scanpy categorical palettes, and restores any palette Scanpy generates during plotting. It's a small wrapper with a deliberately small job. It does not infer an embedding or decide what your data means.

## Roadmap

These are plans, not features hiding behind a very optimistic README.

**v0.1 — Dimensionality reduction**

- Publication-minded embeddings and consistent styling
- Cluster labels and population highlighting
- Improved continuous feature plots
- Automatic rasterization and faceting

**v0.2 — Common single-cell plots**

- Violin plots, boxplots, dotplots, cell-type proportions, and distributions

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

scPolish builds on the Scanpy, AnnData, Matplotlib, Seaborn, NumPy, and pandas ecosystems. Their plotting and data tools do the heavy lifting; scPolish helps them show up to the figure panel looking coordinated.

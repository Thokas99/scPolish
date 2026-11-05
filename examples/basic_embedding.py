"""Generate categorical and continuous embedding examples."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import scpolish as scp

rng = np.random.default_rng(1)
obs = pd.DataFrame(
    {"cell_type": pd.Categorical(["T", "B", "Myeloid"] * 40), "score": rng.random(120)},
    index=[f"cell{i}" for i in range(120)],
)
adata = ad.AnnData(
    rng.poisson(2, (120, 10)),
    obs=obs,
    var=pd.DataFrame(index=[f"gene{i}" for i in range(10)]),
)
adata.obsm["X_umap"] = rng.normal(size=(120, 2))
out = Path("figures")
out.mkdir(exist_ok=True)
for feature in ("cell_type", "score"):
    ax = scp.embedding(adata, color=feature, title=feature)
    ax.figure.savefig(out / f"{feature}.pdf", bbox_inches="tight")
    ax.figure.savefig(out / f"{feature}.png", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(ax.figure)

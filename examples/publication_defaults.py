"""Render PBMC3k plot examples; scores and donor observations are synthetic."""

import pickle
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

import scpolish as scp

out = Path(__file__).resolve().parent / "output" / "plotting_styles"
out.mkdir(parents=True, exist_ok=True)
adata = sc.datasets.pbmc3k_processed()
rng = np.random.default_rng(7)
adata.obs["Demo score A"] = rng.normal(size=adata.n_obs)
adata.obs.loc[adata.obs.index[::5], "Demo score A"] = np.nan
adata.obs["Demo score B"] = rng.normal(size=adata.n_obs)

donors = [f"Donor {i + 1}" for i in range(6)]
donor_obs = pd.DataFrame(
    {
        "donor": np.repeat(donors, 2),
        "condition": ["Control", "Treatment"] * len(donors),
    },
    index=[f"obs{i}" for i in range(12)],
)
donor_obs["Demo value"] = np.repeat(rng.normal(size=6), 2) + np.tile([0, 0.6], 6)
donor_adata = ad.AnnData(obs=donor_obs)
before = adata.copy()
uns_before = pickle.dumps(adata.uns)

plots = {
    "categorical_direct": lambda: scp.dimplot(
        adata, "louvain", labels="direct", palette="pastel"
    ),
    "categorical_numbered": lambda: scp.dimplot(
        adata, "louvain", labels="numbered", legend=True, palette="pastel"
    ),
    "score_single": lambda: scp.featureplot(
        adata, "Demo score A", style="overlay", title="Score A"
    ),
    "scores_horizontal": lambda: scp.featureplot(
        adata, ["Demo score A", "Demo score B"], style="overlay"
    ),
    "violin_standard": lambda: scp.violinplot(
        adata, "MS4A1", "louvain", palette="pastel"
    ),
    "violin_donor_paired": lambda: scp.violinplot(
        donor_adata,
        "Demo value",
        "condition",
        points=True,
        labelby="donor",
        paired_by="donor",
        palette="pastel",
        show_counts=True,
    ),
    "violin_six_features": lambda: scp.violinplot(
        adata,
        ["MS4A1", "CD3D", "NKG7", "FCGR3A", "PPBP", "LST1"],
        "louvain",
        palette="pastel",
    ),
}

for name, plot in plots.items():
    result = plot()
    fig = result.figure if hasattr(result, "figure") else result[0]
    for suffix in ("pdf", "png"):
        with mpl.rc_context({"pdf.fonttype": 42}):
            fig.savefig(
                out / f"{name}.{suffix}",
                dpi=300,
                facecolor="white",
                bbox_inches=None,
            )
    plt.close(fig)

pd.testing.assert_frame_equal(adata.obs, before.obs)
np.testing.assert_array_equal(adata.X, before.X)
np.testing.assert_array_equal(adata.obsm["X_umap"], before.obsm["X_umap"])
assert pickle.dumps(adata.uns) == uns_before
print(out)

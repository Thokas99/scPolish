"""PBMC3k examples; demo scores and donor identities are synthetic (seed 7)."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

import scpolish as scp

out = Path(__file__).resolve().parent / "output" / "publication_defaults"
out.mkdir(parents=True, exist_ok=True)
adata = sc.datasets.pbmc3k_processed()
rng = np.random.default_rng(7)
adata.obs["Demo score A"] = rng.normal(size=adata.n_obs)
adata.obs["Demo score B"] = rng.normal(size=adata.n_obs)
adata.obs.loc[adata.obs.index[::5], "Demo score A"] = np.nan
adata.obs["Score subset A"] = adata.obs["Demo score B"].where(
    np.arange(adata.n_obs) % 2 == 0
)
adata.obs["Score subset B"] = adata.obs["Demo score B"].where(
    np.arange(adata.n_obs) % 2 == 1
)
demo = adata[:18].copy()
demo.obs["Demo group"] = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
demo.obs["Demo donor"] = ["Demo " + str(i) for i in range(6)] * 3
demo.obs["Demo value"] = np.linspace(0, 1, 18)
demo.obs.loc[demo.obs.index[6:12], "Demo value"] = np.nan
adata.obs["Unscored demo"] = np.nan
plots = {
    "all_missing": lambda: scp.featureplot(adata, "Unscored demo"),
    "percentile_limits": lambda: scp.featureplot(
        adata, "Demo score B", vmin="p5", vmax="p95"
    ),
    "paired_missing": lambda: scp.violinplot(
        demo,
        "Demo value",
        "Demo group",
        paired_by="Demo donor",
        points=True,
        show_counts=True,
    ),
    "violin": lambda: scp.violinplot(adata, "MS4A1", "louvain"),
    "violin_points": lambda: scp.violinplot(adata, "MS4A1", "louvain", points=True),
    "violin_grid": lambda: scp.violinplot(
        adata, ["MS4A1", "CD3D", "NKG7", "FCGR3A"], "louvain"
    ),
    "scores_two": lambda: scp.featureplot(adata, ["Demo score A", "Demo score B"]),
    "features_four": lambda: scp.featureplot(
        adata, ["MS4A1", "CD3D", "NKG7", "FCGR3A"]
    ),
    "missing_shared": lambda: scp.featureplot(
        adata, ["Score subset A", "Score subset B"], shared_scale=True
    ),
}
for name, plot in plots.items():
    result = plot()
    fig = result.figure if hasattr(result, "figure") else result[0]
    for suffix in ("pdf", "png"):
        with mpl.rc_context({"pdf.fonttype": 42}):
            fig.savefig(out / f"{name}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
print(out)

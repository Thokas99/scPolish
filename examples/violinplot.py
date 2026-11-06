"""Create real-data examples of scPolish violin plots from PBMC3k."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

import scpolish as scp


def main() -> None:
    adata = sc.datasets.pbmc3k_processed()
    output = Path(__file__).resolve().parent / "output" / "violinplot"
    output.mkdir(parents=True, exist_ok=True)

    print(f"Available annotations: {list(adata.obs.columns)}")
    print(f"Available genes: {adata.n_vars}; examples: {list(adata.var_names[:12])}")

    groupby = next(
        (
            key
            for key in ("louvain", "cell_type", "leiden")
            if key in adata.obs and adata.obs[key].nunique(dropna=True) > 1
        ),
        None,
    )
    if groupby is None:
        raise ValueError("PBMC3k has no suitable categorical group annotation.")

    candidates = ["MS4A1", "CD79A", "S100A8", "NKG7", "FCGR3A", "PPBP"]
    markers = [gene for gene in candidates if gene in adata.var_names]
    print(f"Using grouping {groupby!r} and available marker genes: {markers}")
    if len(markers) < 6:
        raise ValueError(f"Need six valid marker genes; found {len(markers)}.")

    split_by = next(
        (
            key
            for key in ("sample", "sample_id", "batch", "donor", "orig.ident")
            if key in adata.obs
            and key != groupby
            and adata.obs[key].nunique(dropna=True) == 2
        ),
        None,
    )
    synthetic_split = split_by is None
    if split_by is None:
        split_by = "synthetic_split_demo"
        adata.obs[split_by] = np.where(
            np.arange(adata.n_obs) % 2 == 0, "Synthetic A", "Synthetic B"
        )
        print(
            "No suitable split annotation found; using a synthetic parity split "
            "for layout demonstration only."
        )
    else:
        print(f"Using available split annotation {split_by!r}.")

    plots = {
        "violin_standard": {"features": markers[0]},
        "violin_multi": {"features": markers[:4], "ncols": 2},
        "violin_stacked": {"features": markers[:6], "mode": "stacked"},
        "violin_horizontal": {
            "features": markers[0],
            "orientation": "horizontal",
        },
        ("violin_split_synthetic" if synthetic_split else "violin_split"): {
            "features": markers[0],
            "split_by": split_by,
        },
    }
    for name, options in plots.items():
        for extension in ("pdf", "png"):
            result = scp.violinplot(
                adata,
                groupby=groupby,
                save=output / f"{name}.{extension}",
                dpi=300,
                **options,
            )
            fig = result.figure if hasattr(result, "figure") else result[0]
            plt.close(fig)

    print(f"Saved 10 PDF/PNG figures to {output}")


if __name__ == "__main__":
    main()

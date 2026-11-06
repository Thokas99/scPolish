"""Create PBMC3k embedding and feature plot examples."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

import scpolish as scp


def main() -> None:
    adata = sc.datasets.pbmc3k_processed()
    output = Path(__file__).resolve().parent / "output" / "core_plots"
    output.mkdir(parents=True, exist_ok=True)

    groupby = "louvain"
    markers = ["MS4A1", "NKG7", "FCGR3A", "CD79A"]
    missing = [gene for gene in markers if gene not in adata.var_names]
    if groupby not in adata.obs or missing:
        raise ValueError(f"PBMC3k is missing {groupby!r} or genes: {missing}")

    adata.obs["synthetic_split_demo"] = np.where(
        np.arange(adata.n_obs) % 2 == 0, "Synthetic A", "Synthetic B"
    )
    print(f"Using PBMC3k annotations: {list(adata.obs.columns)}")
    print(f"Using available marker genes: {markers}")
    print("The split panels use a synthetic parity label for layout demonstration.")

    examples = {
        "dimplot_louvain": lambda save: scp.dimplot(
            adata, groupby, labels=False, save=save
        ),
        "dimplot_direct_labels": lambda save: scp.dimplot(
            adata, groupby, labels="direct", save=save
        ),
        "dimplot_numbered_labels": lambda save: scp.dimplot(
            adata, groupby, labels="numbered", save=save
        ),
        "dimplot_split_synthetic": lambda save: scp.dimplot(
            adata, groupby, labels=False, split_by="synthetic_split_demo", save=save
        ),
        "featureplot_ms4a1": lambda save: scp.featureplot(adata, "MS4A1", save=save),
        "featureplot_markers": lambda save: scp.featureplot(
            adata, markers, ncols=2, save=save
        ),
        "featureplot_existing_axes": lambda save: _featureplot_on_axes(adata, save),
    }
    for name, plot in examples.items():
        for extension in ("pdf", "png"):
            result = plot(output / f"{name}.{extension}")
            fig = result.figure if hasattr(result, "figure") else result[0]
            plt.close(fig)

    print(f"Saved {2 * len(examples)} PDF/PNG figures to {output}")


def _featureplot_on_axes(adata, save):
    fig, ax = plt.subplots(figsize=(6, 5))
    return scp.featureplot(adata, "MS4A1", ax=ax, save=save)


if __name__ == "__main__":
    main()

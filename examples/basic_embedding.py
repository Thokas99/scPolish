"""Create basic PBMC3k UMAP examples using ``scp.embedding()``."""

from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc

import scpolish as scp


def main() -> None:
    adata = sc.datasets.pbmc3k_processed()
    output = Path(__file__).resolve().parent / "output" / "basic_embedding"
    output.mkdir(parents=True, exist_ok=True)

    examples = [
        ("louvain", "louvain"),
        ("n_genes", "n_genes"),
        ("ms4a1", "MS4A1"),
    ]
    for filename, color in examples:
        if color not in adata.obs and color not in adata.var_names:
            raise ValueError(f"PBMC3k does not contain {color!r}.")
        ax = scp.embedding(adata, color=color, title=color)
        for extension in ("pdf", "png"):
            ax.figure.savefig(output / f"{filename}.{extension}", dpi=300)
        plt.close(ax.figure)

    print(f"Saved {2 * len(examples)} PDF/PNG figures to {output}")


if __name__ == "__main__":
    main()

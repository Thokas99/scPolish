import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from scpolish import embedding


@pytest.mark.parametrize("color", ["cell_type", "score", "gene0"])
def test_embedding_colors(adata, color, tmp_path):
    before = adata.copy()
    ax = embedding(adata, color=color)
    assert isinstance(ax, Axes)
    assert ax.collections
    assert ax.collections[0].get_rasterized()
    ax.figure.savefig(tmp_path / "plot.png")
    assert (tmp_path / "plot.png").stat().st_size > 0
    assert np.array_equal(adata.X, before.X)
    assert list(adata.uns) == list(before.uns)
    plt.close(ax.figure)


def test_existing_axes(adata):
    _, expected = plt.subplots()
    assert embedding(adata, ax=expected) is expected
    plt.close(expected.figure)


def test_missing_basis_is_informative(adata):
    with pytest.raises(ValueError, match="Available.*umap"):
        embedding(adata, basis="missing")


def test_scanpy_generated_palette_is_not_retained(adata):
    embedding(adata, color="cell_type")
    assert "cell_type_colors" not in adata.uns
    plt.close("all")


def test_existing_scanpy_palette_is_preserved(adata):
    adata.uns["cell_type_colors"] = ["#112233", "#445566", "#778899"]
    palette = adata.uns["cell_type_colors"].copy()
    ax = embedding(adata, color="cell_type")
    assert adata.uns["cell_type_colors"] == palette
    plt.close(ax.figure)

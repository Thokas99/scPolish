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
    assert ax.get_aspect() == 1.0
    assert ax.collections
    assert not any(collection.get_rasterized() for collection in ax.collections)
    ax.figure.savefig(tmp_path / "plot.png")
    assert (tmp_path / "plot.png").stat().st_size > 0
    assert np.array_equal(adata.X, before.X)
    assert list(adata.uns) == list(before.uns)
    plt.close(ax.figure)


def test_embedding_rasterization_is_opt_in(adata):
    ax = embedding(adata, color="score", raster=True)
    assert all(collection.get_rasterized() for collection in ax.collections)
    plt.close(ax.figure)


def test_existing_axes(adata):
    _, expected = plt.subplots()
    assert embedding(adata, ax=expected) is expected
    assert expected.get_aspect() == "auto"
    plt.close(expected.figure)


def test_custom_embedding_canvas_fills_panel(adata):
    ax = embedding(adata, color="score", figsize=(12, 8))
    assert ax.get_aspect() == "auto"
    assert tuple(ax.figure.get_size_inches()) == (12, 8)
    plt.close(ax.figure)


def test_embedding_uses_adaptive_point_area_and_honors_override(adata):
    from scpolish._utils.style import auto_point_size

    ax = embedding(adata, color="score")
    np.testing.assert_allclose(
        ax.collections[0].get_sizes(), auto_point_size(adata.n_obs)
    )
    plt.close(ax.figure)
    ax = embedding(adata, color="score", size=7.0)
    np.testing.assert_allclose(ax.collections[0].get_sizes(), 7.0)


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

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def adata():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 5))
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(["A", "B", "C"] * 10),
            "score": np.linspace(0, 1, 30),
        },
        index=[f"cell{i}" for i in range(30)],
    )
    obj = ad.AnnData(x, obs=obs, var=pd.DataFrame(index=[f"gene{i}" for i in range(5)]))
    obj.obsm["X_umap"] = rng.normal(size=(30, 2))
    return obj

"""Small compatibility helpers for Scanpy's evolving public API."""

from inspect import signature


def obs_df(adata, keys, *, layer=None, use_raw=False, use=None):
    """Read plotted values with Scanpy's current or legacy source arguments."""
    if not isinstance(use_raw, bool):
        raise ValueError("use_raw must be a boolean.")
    import scanpy as sc

    supports_use = "use" in signature(sc.get.obs_df).parameters
    if use is not None:
        if not supports_use:
            raise ValueError("The 'use' argument requires Scanpy 1.13 or later.")
        if layer is not None or use_raw:
            raise ValueError("use cannot be combined with layer or use_raw.")
        return sc.get.obs_df(adata, keys=keys, use=use)
    if supports_use:
        if layer is not None:
            from anndata.acc import A

            return sc.get.obs_df(adata, keys=keys, use=A.layers[layer])
        if use_raw:
            # shortcut: Scanpy 1.13 has no raw accessor; revisit if use_raw is removed.
            return sc.get.obs_df(adata, keys=keys, use_raw=True)
        return sc.get.obs_df(adata, keys=keys)
    return sc.get.obs_df(adata, keys=keys, layer=layer, use_raw=use_raw)


def matplotlib_plotting(sc):
    """Return the Matplotlib-backed API under Scanpy's preset-based backends."""
    return getattr(sc.pl, "legacy", sc.pl)

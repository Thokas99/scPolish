"""Categorical palette generation and validation."""

from collections.abc import Mapping, Sequence
from functools import lru_cache
from warnings import warn

import glasbey
import pandas as pd
from anndata import AnnData
from matplotlib.colors import to_hex, to_rgba

PALETTES = {
    "muted": ((40, 80), (20, 65)),
    "pastel": ((65, 90), (15, 50)),
}


def categorical_categories(values: pd.Series) -> list:
    """Return observed categories in categorical or deterministic order."""
    observed = set(values.dropna().unique())
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [category for category in values.cat.categories if category in observed]
    return sorted(observed, key=lambda value: (str(type(value)), str(value)))


def validate_colors(colors: Sequence) -> tuple[str, ...]:
    """Normalize valid Matplotlib colors to hexadecimal strings."""
    try:
        return tuple(to_hex(to_rgba(color)) for color in colors)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Matplotlib color: {exc}") from exc


@lru_cache(maxsize=128)
def _glasbey_palette(preset: str, count: int) -> tuple[str, ...]:
    lightness, chroma = PALETTES[preset]
    colors = validate_colors(
        glasbey.create_palette(
            palette_size=count,
            lightness_bounds=lightness,
            chroma_bounds=chroma,
        )
    )
    if len(colors) != count or len(set(colors)) != count:
        raise RuntimeError(f"Glasbey did not produce {count} distinct colors.")
    return colors


def _scanpy_palette(adata: AnnData, key: str, categories: list) -> dict:
    """Reuse stored Scanpy colors or ask Scanpy for its categorical defaults."""
    values = adata.obs[key]
    levels = (
        list(values.cat.categories)
        if isinstance(values.dtype, pd.CategoricalDtype)
        else categorical_categories(values)
    )
    stored = adata.uns.get(f"{key}_colors")
    if stored is not None and len(stored) == len(levels):
        try:
            stored_colors = validate_colors(stored)
        except ValueError:
            stored_colors = None
        if stored_colors is not None:
            mapping = dict(zip(levels, stored_colors, strict=True))
            return {category: mapping[category] for category in categories}

    obj = AnnData(obs=adata.obs[[key]].copy())
    obj.obs[key] = pd.Categorical(obj.obs[key], categories=levels)
    from scanpy.plotting._tools.scatterplots import _get_palette

    colors = _get_palette(obj, key, palette=None)
    return {category: to_hex(to_rgba(colors[category])) for category in categories}


def _fallback(categories: list, reason: str) -> dict:
    """Warn and replace an invalid custom palette in full."""
    count = len(categories)
    warn(
        f"{reason} Falling back to the {count}-color muted Glasbey palette.",
        UserWarning,
        stacklevel=3,
    )
    return dict(zip(categories, _glasbey_palette("muted", count), strict=True))


def resolve_palette(
    categories: Sequence,
    palette="muted",
    *,
    adata: AnnData | None = None,
    key: str | None = None,
) -> dict:
    """Resolve one deterministic color for each category in the given order."""
    categories = list(categories)
    if not categories:
        return {}
    if len(set(categories)) != len(categories):
        raise ValueError("categories must be unique.")

    if palette is None:
        palette = "scanpy"
    if isinstance(palette, str):
        if palette in PALETTES:
            return dict(
                zip(categories, _glasbey_palette(palette, len(categories)), strict=True)
            )
        if palette == "scanpy":
            if adata is None or key is None or key not in adata.obs:
                raise ValueError("palette='scanpy' requires an AnnData key.")
            return _scanpy_palette(adata, key, categories)
        raise ValueError(
            f"Unknown palette preset {palette!r}; choose 'muted', 'pastel', "
            "or 'scanpy'."
        )

    if isinstance(palette, Mapping):
        missing = [category for category in categories if category not in palette]
        unknown = [category for category in palette if category not in set(categories)]
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing categories {missing!r}")
            if unknown:
                details.append(f"unknown categories {unknown!r}")
            return _fallback(
                categories,
                "Custom palette dictionary has " + " and ".join(details) + ".",
            )
        values = [palette[category] for category in categories]
    elif isinstance(palette, Sequence):
        values = list(palette)
        if len(values) != len(categories):
            return _fallback(
                categories,
                f"Custom palette contains {len(values)} colors, but "
                f"{len(categories)} categories were detected.",
            )
    else:
        return _fallback(
            categories,
            f"Unsupported custom palette type {type(palette).__name__!r}.",
        )

    try:
        colors = validate_colors(values)
    except ValueError as exc:
        return _fallback(
            categories, f"Custom palette contains an invalid color ({exc})."
        )
    if len(set(colors)) != len(colors):
        return _fallback(
            categories, "Custom palette assigns duplicate colors to categories."
        )
    return dict(zip(categories, colors, strict=True))


def categorical_colors(
    adata: AnnData, key: str, categories: Sequence, palette=None
) -> tuple[str, ...]:
    """Compatibility helper returning resolved colors in the supplied order."""
    mapping = resolve_palette(categories, palette, adata=adata, key=key)
    return tuple(mapping[category] for category in categories)

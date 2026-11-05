"""Small color helpers used by plotting functions."""

from collections.abc import Sequence

from matplotlib.colors import is_color_like


def validate_colors(colors: Sequence[str]) -> tuple[str, ...]:
    """Return valid Matplotlib colors as a tuple."""
    result = tuple(colors)
    invalid = [color for color in result if not is_color_like(color)]
    if invalid:
        raise ValueError(f"Invalid Matplotlib color(s): {invalid}")
    return result


def categorical_colors(adata, key: str, n_categories: int) -> tuple[str, ...] | None:
    """Get Scanpy's stored colors or a restrained default palette."""
    stored = adata.uns.get(f"{key}_colors")
    if stored is not None:
        colors = validate_colors(stored)
        if len(colors) >= n_categories:
            return colors[:n_categories]
    if n_categories == 0:
        return ()
    from matplotlib import colormaps

    return tuple(
        colormaps["tab20"](i / max(n_categories - 1, 1)) for i in range(n_categories)
    )

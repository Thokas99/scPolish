# Violin style refinement

## Intent

Bring `scpolish.violinplot()` closer to the supplied ggpubr and SCpubr examples: filled violins with clear dark outlines, a narrow white box overlay, and optional dark observation points. Keep the plots readable when labels and paired observations are requested.

## Goals and constraints

- Reuse the existing Seaborn and Matplotlib renderer, muted/pastel palette, scoped style context, layout and export helpers.
- Set a near-black violin outline with a restrained line width; keep `inner=None`, `cut=0`, and the white box overlay at width `0.22`.
- Keep points opt-in, dark grey, and area 18. Preserve observation labels at fontsize 6 with `adjustText`, paired lines, and observation counts.
- Allow optional `statannotations` brackets from caller-supplied p-values; never run tests automatically.
- Keep split-condition summaries visually distinct when the box overlay is enabled.
- Preserve AnnData, avoid global style changes, and do not add tests for statistical inference or aggregate cells into donors.

## Non-goals

- No new plotting framework, automatic statistical tests, donor aggregation, or standalone box/strip plot API.
- No automatic point display or automatic change of violin behavior for small groups.
- No change to categorical palette selection or feature/embedding plot behavior.

## Optional annotations

The optional `annotations` extra adds Seaborn pair brackets from explicit p-values. It does not run tests; one p-value is required for each pair and each feature panel.

## Approaches

1. **Refine the current Seaborn layers (selected):** use a dark violin edge, retain the white box and optional points, and improve label/paired-split alignment. This matches the supplied filled-violin examples with the smallest API and code change.
2. Replace Seaborn violins with custom Matplotlib shapes and layer positions. This offers more control but duplicates rendering behavior already provided by Seaborn.
3. Add named ggpubr/SCpubr style presets. This adds a new public API and more combinations without a need established by the examples.

## Implementation scope

Update `_violinplot.py`, its regression tests and optional dependency, the PBMC3k publication example, and the README. Export fresh PDF/PNG examples and inspect the filled violin, split, points, labels, and pairing layers for contrast, overlap, and clipping.

## Calls left to the implementer

Use the closest existing palette and a near-black outline; retain points as optional. Keep data handling and returned Matplotlib axes unchanged.

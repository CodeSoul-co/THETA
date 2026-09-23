# Visual language for adapted and new charts

Use the user's requested style first. Otherwise preserve the selected template's existing code, composition and visual settings. Apply this guide to necessary local edits; it is not an invitation to redesign a working template. New compositions are reserved for substantial new information that cannot fit through local edits or for explicit user requests. This is a reusable visual language, not an official Nature style standard.

## Color

Keep the selected template's palette and retained categories' color assignments by default. Extend it only for added categories or a necessary encoding change. Available palettes are:

| Starting point | Colors | Typical role |
|---|---|---|
| Figure 07 | `#383838` · `#e8a625` · `#58a3ad` | Neutral control plus warm/cool contrasts |
| Figure 04 | `#7297d5` · `#e49386` · `#459b9e` · `#ece23e` | Four series; use dark outlines when pale colors have low contrast |
| Figure 05 | `#8aaa49` · `#b26462` · `#c8ac32` · `#8c8c8b` | Muted distributions with a neutral comparison |
| Figure 12 | `#edf4e7` → `#9ac87d` → `#4c9a48` → `#276335` | Ordered, low-to-high sequential values |
| Figure 19 | `#81b2c4` · `#8cbbb0` | Soft fills with dark text and structural lines |

Retain category-to-color mapping across panels. Use a sequential palette for magnitude, a diverging palette only for a meaningful midpoint, and discrete colors for categories. Do not cycle colors ambiguously when groups outgrow a palette; add direct labels or facets. Distinguish groups by marker/line shape as needed. Read the selected `style.json` and `plot.py`: some encodings and colormaps live in code.

## Structure and density

Start with a white background, restrained spines, sparse purposeful ticks, and dark labels. The examples combine observations with a summary (points + bars, distributions + boxes) and align complementary panels to save repeated labels. Keep shared category order and scales wherever comparisons require them.

Preserve the selected template's typography, line widths, axes rectangles, panel order and canvas proportions. Removing a missing-data layer does not justify moving unrelated panels. If a removed panel leaves an unusable gap, reclaim space locally while retaining the remaining arrangement. Canvas coordinates in `context.ax([left, top, width, height])` are pixels; Matplotlib text and line sizes are point-based.

Maintain enough space for labels, units, legends and colorbars. Resolve overlap locally with spacing, label wrapping or marker adjustments before enlarging the canvas or creating facets. Aggregation or binning needs a stated rule. Do not silently drop observations or fill missing data to mimic a reference density. Change the layout only for a demonstrated readability/encoding problem, substantial new information, or a user request.

## Quantitative encodings

- Bars normally start at zero. Any break or truncated range must be visually explicit and appropriate.
- Bubble **area** should encode the size metric; Matplotlib scatter `s` is already an area in points squared. Explain units and show a meaningful size legend.
- Show SD, SEM or CI only with an explicit definition and appropriate calculation. Do not invent independent replicates.
- Density bandwidth and normalization affect comparison; document both. The bundled ridge defaults normalize peaks, so height is not sample count.
- Use paired lines only for actual pairings; use a common scale when comparing panels quantitatively.
- Show statistical annotations only from the current analysis, with enough method context to interpret them.

## Final inspection

Compare the rendered composition against the selected preview and account for each layout change. Inspect at the intended reading size, then zoom to investigate clipping or overlaps. Check the smallest labels, longest category names, extreme values, color/size legends, and any inset. Removed layers must leave no stale legend entries or statistical text. Verify export dimensions; inspect SVG/PDF when vector editing or print use matters. Deliver the adapted code, its template origin and a documented data mapping.

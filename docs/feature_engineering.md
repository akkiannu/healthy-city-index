# Feature Engineering Notes

This document captures the transformations we apply on top of raw data sources before displaying indicators, computing the Healthy City Index (HCI), or generating reports.

## Scoring and normalisation

| Feature | Transformation | Formula | Purpose |
| --- | --- | --- | --- |
| Air pollution score | Combination of PM2.5 and NO₂ with inverse min-max scaling. | \\(0.5 	imes rac{100 - (\mathrm{PM}_{2.5}-10)}{90} + 0.5 	imes rac{80 - (\mathrm{NO}_2-5)}{75}\\) (clamped 0–1) | Penalises both particulate matter and NO₂ exceedances. |
| Water pollution score | SWIR turbidity ratio with inverse min-max scaling. | \\(1 - rac{	ext{SWIR} - 0.85}{1.30 - 0.85}\\) (clamped 0–1) | Flags higher turbidity / pollution. |
| Green score | NDVI scaled between 0.1 and 0.8. | \\(rac{	ext{NDVI} - 0.1}{0.8 - 0.1}\\) (clamped 0–1) | Captures vegetation vigour. |
| Built score | NDBI scaled between −0.1 and 0.6, inverted. | \\(1 - rac{	ext{NDBI} + 0.1}{0.6 + 0.1}\\) (clamped 0–1) | Rewards softer, less impervious surfaces. |
| Composite HCI | Weighted blend of pillar scores. | \\(0.28 	imes s_{air} + 0.22 	imes s_{water} + 0.25 	imes s_{green} + 0.25 	imes s_{built}\\) | Produces a single headline number for comparisons. |

All min/max helpers clamp inputs to defensively handle missing or extreme values.

## Earth Engine value handling

- **Built index heatmap**: When sampling Sentinel‑2 derived NDBI through Earth Engine, we fill any `NaN` tiles with the median value of the sampled grid before constructing the heatmap. This avoids hard jumps when the AOI lacks coverage.
- **Vegetation overrides**: If Earth Engine fails to return NDBI for the selected point, we reuse the heatmap median to keep the point metrics consistent with the map.

## Local raster sampling

- `hci_app/raster.py` samples GeoTIFFs and, when a pixel is nodata, searches outward (up to radius 10) to find the nearest valid value.
- Heatmap DataFrames are normalised to [0,1], converted into RGB through the palette, and include preformatted display strings.

## Scenario simulator (`hci_app/simulator.py`)

Our what-if tool adjusts the baseline `RegionData` with a linear response model:

- Increasing NDVI cools land-surface temperature and improves air, water, NDWI, and SAVI via configurable coefficients.
- Increasing NDBI introduces the opposite deltas (warmer, more polluted) and nudges NDWI/SAVI downward.
- All derived values are clamped to realistic ranges before rounding.

These scenario metrics feed into the HCI scoring pipeline and the PDF report.

## AI report assembly (`hci_app/report.py`)

- The PDF enumerates current conditions, scenario summary, and the AI recommendation text. It embeds a radar chart rendered with matplotlib and a simple Mumbai extent map annotated with NDVI/NDBI targets.
- Text content is wrapped to 90 characters per line to avoid overflow.

## Other adjustments

- Heatmap legends are generated on the fly with HTML/CSS snippets for consistent styling.
- Source notes list which metrics are live (Earth Engine / APIs) versus mocked, based on whether real data was returned during the session.


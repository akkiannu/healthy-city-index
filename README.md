# healthy-city-index

Healthy City Index for NASA Space Apps Challenge 2025: Using NASA Earth observation data to identify urban areas needing environmental and health improvements.

## Streamlit prototype (Python-first)

The `streamlit_app.py` script mirrors the React wireframe using Streamlit, giving you a Python-heavy surface for quick iteration.

### Run locally

1. (Optional) create and activate a virtual environment
2. `pip install -r requirements.txt`
3. `streamlit run streamlit_app.py`

## Required data & credentials

| Component | What you need | Default expectation |
| --- | --- | --- |
| Population density raster | GeoTIFF (e.g. WorldPop) placed in `data/` | `ind_pd_2020_1km_UNadj.tif` if available; otherwise provide a path in the sidebar. |
| Population total raster | GeoTIFF with total counts | `ind_pop_2025_CN_1km_R2025A_UA_v1.tif` if present. |
| Tree cover raster | Hansen (or similar) canopy tiles in `data/treecover/` | Any tile that covers Mumbai; the app detects filenames such as `10N_070E.tif`. |
| Google Earth Engine credentials | Service-account JSON | Supply a file and point the sidebar input to it. Default path `credentials.json`. |
| OpenAI API key *(optional)* | `OPENAI_API_KEY` environment variable | Enables the AI recommendation text and PDF section.

Without the rasters the app falls back to mock values; without Earth Engine credentials vegetation/water/built metrics revert to placeholders. NASA POWER and Open-Meteo are accessed anonymously.

Optional: drop the rasters above into `data/`—the app auto-fills
`ind_pd_2020_1km_UNadj.tif` (population density) and
`ind_pop_2025_CN_1km_R2025A_UA_v1.tif` (population totals) when present. Tree
cover tiles in `data/treecover/` (e.g. `10N_070E.tif`) are also detected
automatically. You can point to custom rasters via the sidebar fields. Other
indicators remain mocked until their data sources are wired.

See `docs/data_sources.md` for a full breakdown of inputs and `docs/feature_engineering.md` for the derived metrics roadmap.

Use the “Heatmap overlay” selector in the sidebar to visualise population
density, total population, or tree cover across the Mumbai extent; hover over
any cell to see the sampled value.

The Streamlit UI exposes the same mocked indicators, scoring logic, and recommendations. The current map view uses a blank pydeck canvas with a highlighted industrial rectangle and point marker. See the roadmap at the foot of the app for ideas on layering cached tiles or NASA rasters as we iterate on the mapping experience.

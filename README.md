# healthy-city-index

Healthy City Index for NASA Space Apps Challenge 2025: Using NASA Earth observation data to identify urban areas needing environmental and health improvements.

## Streamlit prototype (Python-first)

The `streamlit_app.py` script mirrors the React wireframe using Streamlit, giving you a Python-heavy surface for quick iteration.

### Run locally

1. (Optional) create and activate a virtual environment
2. `pip install -r requirements.txt`
3. `streamlit run streamlit_app.py`

Optional: drop WorldPop (or similar) GeoTIFFs inside `data/`—the app auto-fills
`ind_pd_2020_1km_UNadj.tif` (population density) and
`ind_pop_2025_CN_1km_R2025A_UA_v1.tif` (population totals) when present. Tree
cover tiles in `data/treecover/` (e.g. `10N_070E.tif`) are also detected
automatically. You can point to custom rasters via the sidebar fields. Other
indicators remain mocked until their data sources are wired.

Use the “Heatmap overlay” selector in the sidebar to visualise population
density, total population, or tree cover across the Mumbai extent; hover over
any cell to see the sampled value.

The Streamlit UI exposes the same mocked indicators, scoring logic, and recommendations. The current map view uses a blank pydeck canvas with a highlighted industrial rectangle and point marker. See the roadmap at the foot of the app for ideas on layering cached tiles or NASA rasters as we iterate on the mapping experience.

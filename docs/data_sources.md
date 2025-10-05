# Healthy City Index — Data Sources

This project blends local rasters, NASA resources, and other open APIs to populate the Mumbai-focused Healthy City Index. The table below documents each source, why it is used, and key implementation notes.

| Dataset / API | Purpose in app | Access path | Notes |
| --- | --- | --- | --- |
| **WorldPop / population rasters** (e.g. `ind_pd_2020_1km_UNadj.tif`, `ind_pop_2025_CN_1km_R2025A_UA_v1.tif`) | Provides population density and total population heatmaps and point samples. | Optional local files under `data/`; paths can be set in the sidebar. | Any GeoTIFF with population density/total counts can be supplied. When missing, the UI falls back to mock values. |
| **Hansen Global Forest Change / tree-cover tiles** (e.g. `data/treecover/10N_070E.tif`) | Supplies canopy percentage heatmap and per-point samples. | Optional local files under `data/treecover/`; path configurable via sidebar. | The app expects percentage canopy. Absent data triggers mock values. |
| **Google Earth Engine – Sentinel‑2 SR** | Computes NDVI, NDWI, NDBI, SAVI per selection; also powers "Built index" heatmap. | Queried dynamically via service-account credentials (defaults to `credentials.json`). | Requires provisioning a Google Cloud project and Earth Engine service account. See README for credential setup. |
| **Google Earth Engine – Sentinel/Landsat SWIR ratio** | Drives coastal water pollution heatmap and per-point proxy. | Same credential flow as above. | The helper falls back to mock data when Earth Engine is unavailable. |
| **Google Earth Engine – MODIS/061/MOD11A2** | Produces the land-surface-temperature heatmap. | Same credential flow. | Temperature values are resampled to the Mumbai bounds and converted to °C. |
| **NASA POWER API** (`TS` parameter) | Supplies point-based land-surface temperature when Earth Engine fails or credentials are absent. | HTTPS request from the Streamlit runtime. | No authentication required; we average the last seven days of skin temperature. |
| **Open-Meteo Air Quality API** | Populates PM2.5, NO₂, CO, CO₂ metrics and feeds the air score. | HTTPS request from the Streamlit runtime. | Values reflect the last 24 hours around the selected coordinate. |
| **Local industrial proximity shapefile** *(future)* | Placeholder mentioned in UI for future enhancements. | Not yet wired. | Historical TODO in the roadmap. |
| **OpenAI API** | Generates the AI action plan embedded in the downloadable report. | Requires `OPENAI_API_KEY` environment variable. | Optional; if the key is absent the UI shows a tip and report includes a placeholder message. |

### Credential summary

- **Google Earth Engine**: Provide a service-account JSON (default path `credentials.json`). Used for vegetation indices, water pollution, land-surface-temperature heatmaps, and built index.
- **OpenAI API**: Needed only if you want the AI recommendation section to be populated. Export `OPENAI_API_KEY` before launching Streamlit.
- **NASA POWER / Open-Meteo**: No credentials required.

### Local data folders

- Drop optional rasters inside `data/` or `data/treecover/`. The app auto-detects filenames and you can override via the sidebar inputs.


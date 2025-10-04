# healthy-city-index

Healthy City Index for NASA Space Apps Challenge 2025: Using NASA Earth observation data to identify urban areas needing environmental and health improvements.

## Frontend (Vite + React + TypeScript)

The new `frontend/` folder hosts a self-contained React application that renders the Healthy City Index explorer for Mumbai. It includes a mock data generator, indicator visualisations (Recharts radar plot), an interactive Leaflet map (tiles disabled, so no network prompts), and recommendation panels ready to be wired to the Python backend.

### Getting started

1. `cd frontend`
2. `npm install`
3. `npm run dev`

Open the URL that Vite prints (default http://localhost:5173) to preview the interface. The mock data should update as you click anywhere on the blank Leaflet canvas or edit the latitude/longitude fields.

### Next steps

- Replace the placeholder `fetchRegionData` function with an API call to your backend endpoint (suggested route: `/api/region?lat=&lon=`).
- Add loading/error states for the API call.
- Optional: enable a basemap by adding a `TileLayer` to the map once network access is approved.
- Consider populating the AI tab with responses from your selected LLM once the backend integration is ready.

## Streamlit prototype (Python-first)

The `streamlit_app.py` script mirrors the React wireframe using Streamlit, giving you a Python-heavy surface for quick iteration.

### Run locally

1. (Optional) create and activate a virtual environment
2. `pip install -r requirements.txt`
3. `streamlit run streamlit_app.py`

The Streamlit UI exposes the same mocked indicators, scoring logic, and recommendations. The current map view uses a blank pydeck canvas with a highlighted industrial rectangle and point marker. See the roadmap at the foot of the app for ideas on layering cached tiles or NASA rasters as we iterate on the mapping experience.

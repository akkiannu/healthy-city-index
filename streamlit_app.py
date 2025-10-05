"""Streamlit version of the Healthy City Index wireframe.

This script mirrors the existing React prototype, providing a Python-first
implementation with mocked indicator data, scoring, and recommendations. The
map section uses pydeck with a blank base layer so the app remains functional
without pulling remote tiles. Future iterations can swap in cached rasters or
custom vector layers once NASA data products are ready.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hci_app.constants import (
    DEFAULT_POINT,
    DEFAULT_POP_DENSITY_RASTER,
    DEFAULT_POP_TOTAL_RASTER,
    DEFAULT_TREECOVER_RASTER,
    HEATMAP_COLOR_SCHEMES,
    MUMBAI_BOUNDS,
    PILLAR_DISPLAY,
)
from hci_app.airquality import fetch_air_quality
from hci_app.earthengine import (
    DEFAULT_CREDENTIALS_PATH,
    EarthEngineUnavailable,
    vegetation_indices as gee_vegetation_indices,
    built_index_heatmap as gee_built_index_heatmap,
    lst_at_point as gee_lst_at_point,
    lst_heatmap as gee_lst_heatmap,
    water_pollution_heatmap as gee_water_pollution_heatmap,
    water_pollution_indices as gee_water_pollution,
)
from hci_app.nasa import fetch_lst
from hci_app.maps import map_layers, prepare_heatmap_dataframe, selection_to_point
from hci_app.models import RegionData, fetch_region_data
from hci_app.raster import raster_dataframe, raster_value_from_path
from hci_app.scoring import clamp_value, compute_scores, recommendations
from hci_app.llm import LLMUnavailable, generate_plan
from hci_app.report import generate_pdf_report, map_snapshot_png, radar_chart_png
from hci_app.simulator import simulate_urban_scenario


st.set_page_config(page_title="Healthy City Index — Mumbai", layout="wide")


METRIC_IMPACTS: Dict[str, str] = {
    "PM2.5": "High values flag urgent air-quality controls.",
    "NO₂ (column)": "Tracks traffic pressure and ventilation gaps.",
    "CO (column)": "Points to combustion hotspots to fix.",
    "CO₂": "Signals carbon-intensive districts to decarbonise.",
    "Water pollution (SWIR ratio)": "Shows where coastal water runs murky.",
    "NDVI": "Reveals how much cooling greenery exists.",
    "NDWI": "Marks wetter ground or flood buffers.",
    "NDBI": "Highlights hardscape-heavy blocks.",
    "SAVI": "Flags stressed urban vegetation.",
    "Population density": "Dense spots need upgraded services.",
    "Population (est.)": "Total residents guide capacity sizing.",
    "Tree cover": "Canopy cools streets—protect high areas.",
    "Land surface temp": "Hotter tiles are heat-island targets.",
    "Industrial proximity": "Nearby plants need buffers and zoning care.",
    "HCI (0–1, higher better)": "Overall livability snapshot.",
    "Composite HCI": "Quick resilience headline for planners.",
    "Latitude": "Fixes the site for cross-referencing maps.",
    "Longitude": "Pairs with latitude to locate the block.",
    "LST": "Surface heat needing shade or cool roofs.",
}

BASEMAP_TEMPLATES: Dict[str, str] = {
    "Street view": "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "Satellite view": "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Terrain view": "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
}


def _initialise_session_state() -> None:
    if "point" not in st.session_state:
        st.session_state.point = DEFAULT_POINT.copy()


def _radar_chart(scores: Dict[str, float]) -> go.Figure:
    pillar_keys = ["air", "water", "green", "built"]
    categories = [PILLAR_DISPLAY[key] for key in pillar_keys]
    values = [round(scores[key], 3) * 100 for key in pillar_keys]
    return go.Figure(
        data=go.Scatterpolar(
            r=values + values[:1], theta=categories + categories[:1], fill="toself"
        )
    ).update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
        showlegend=False,
        autosize=True,
        margin=dict(l=20, r=20, t=20, b=20),
    )


def _heatmap_legend_html(
    name: str, colors: list[list[int]], legend_text: Optional[str]
) -> Optional[str]:
    if not colors:
        return None
    stops = []
    total = max(len(colors) - 1, 1)
    for idx, (r, g, b) in enumerate(colors):
        percent = idx / total * 100
        stops.append(f"#{r:02x}{g:02x}{b:02x} {percent:.0f}%")

    low_text = "Low"
    high_text = "High"
    if legend_text:
        parts = [part.strip() for part in legend_text.split("·")]
        if parts:
            low_text = parts[0]
            if len(parts) > 1:
                high_text = parts[1]

    gradient = ", ".join(stops)
    return f"""
<style>
.hci-legend {{
  display: grid;
  grid-template-columns: 1fr;
  row-gap: 0.25rem;
  font-size: 0.7rem;
  margin-top: 0.35rem;
  margin-bottom: 0.6rem;
}}
.hci-legend-title {{
  font-weight: 600;
  text-align: center;
  letter-spacing: 0.04em;
}}
.hci-legend-bar-row {{
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  column-gap: 0.5rem;
}}
.hci-legend-bar {{
  height: 12px;
  border-radius: 999px;
  background: linear-gradient(90deg, {gradient});
  border: 1px solid rgba(15, 23, 42, 0.08);
}}
.hci-legend-text {{
  color: rgba(15, 23, 42, 0.7);
}}
</style>
<div class="hci-legend">
  <div class="hci-legend-title">{name}</div>
  <div class="hci-legend-bar-row">
    <span class="hci-legend-text">{low_text}</span>
    <div class="hci-legend-bar"></div>
    <span class="hci-legend-text">{high_text}</span>
</div>
</div>
"""




def main() -> None:
    _initialise_session_state()
    st.session_state.setdefault("llm_response", None)
    st.session_state.setdefault("llm_error", None)
    st.session_state.setdefault("llm_error", None)

    default_density_path = (
        str(DEFAULT_POP_DENSITY_RASTER.resolve())
        if DEFAULT_POP_DENSITY_RASTER.exists()
        else ""
    )
    default_population_path = (
        str(DEFAULT_POP_TOTAL_RASTER.resolve())
        if DEFAULT_POP_TOTAL_RASTER.exists()
        else ""
    )
    default_treecover_path = (
        str(DEFAULT_TREECOVER_RASTER.resolve())
        if DEFAULT_TREECOVER_RASTER and DEFAULT_TREECOVER_RASTER.exists()
        else ""
    )

    st.session_state.setdefault("population_raster_path", default_density_path)
    st.session_state.setdefault("population_total_raster_path", default_population_path)
    st.session_state.setdefault("treecover_raster_path", default_treecover_path)

    st.sidebar.header("Data sources")
    population_raster_input = st.sidebar.text_input(
        "Population density raster (*.tif)",
        value=st.session_state["population_raster_path"],
        help=(
            "Provide the path to a WorldPop (or similar) density GeoTIFF. The"
            " app samples the raster at the selected coordinate to replace the"
            " mock population density."
        ),
    )
    st.session_state["population_raster_path"] = population_raster_input

    population_total_input = st.sidebar.text_input(
        "Population total raster (*.tif)",
        value=st.session_state["population_total_raster_path"],
        help=(
            "Optional WorldPop total population raster for the same region."
            " When provided, the sampled population count is displayed alongside"
            " density."
        ),
    )
    st.session_state["population_total_raster_path"] = population_total_input

    treecover_input = st.sidebar.text_input(
        "Tree cover raster (*.tif)",
        value=st.session_state["treecover_raster_path"],
        help="Optional tree cover GeoTIFF (e.g. Hansen Global Forest Change tiles).",
    )
    st.session_state["treecover_raster_path"] = treecover_input

    population_status_placeholder = st.sidebar.empty()
    population_total_status_placeholder = st.sidebar.empty()
    treecover_status_placeholder = st.sidebar.empty()
    if not population_raster_input:
        population_status_placeholder.info("Density: mock values in use.")
    if not population_total_input:
        population_total_status_placeholder.info("Totals: mock values in use.")
    if not treecover_input:
        treecover_status_placeholder.info("Tree cover: mock values in use.")

    st.sidebar.header("Remote data")
    st.session_state.setdefault("use_gee", False)
    st.session_state.setdefault("gee_credentials_path", str(DEFAULT_CREDENTIALS_PATH))
    use_gee = st.sidebar.checkbox(
        "Use Google Earth Engine vegetation indices",
        value=st.session_state["use_gee"],
        help="Fetch NDVI/NDWI/NDBI/SAVI from Sentinel-2 via Google Earth Engine.",
    )
    st.session_state["use_gee"] = use_gee
    gee_credentials_input = st.sidebar.text_input(
        "Earth Engine credentials JSON",
        value=st.session_state["gee_credentials_path"],
        help="Path to the service-account JSON used for Earth Engine authentication.",
    )
    st.session_state["gee_credentials_path"] = gee_credentials_input
    gee_credentials_resolved: Optional[Path] = None
    if gee_credentials_input:
        candidate = Path(gee_credentials_input).expanduser()
        if candidate.exists():
            gee_credentials_resolved = candidate
        elif use_gee:
            st.sidebar.warning(
                "Earth Engine credentials not found; vegetation metrics will use mock values."
            )

    st.title("🌆 Healthy City Index — Mumbai")
    st.caption("Streamlit port • Mock indicators that mirror the React wireframe")

    col_header, col_repo = st.columns([4, 1])
    with col_repo:
        st.markdown(
            "[GitHub Repo](https://github.com/akkiannu/healthy-city-index)"
            " ↗",
            help="Open the project repository in a new tab.",
        )

    tab_map, tab_ai = st.tabs(["🗺️ Map Explorer", "🧠 AI Recommendations"])

    mock_tracker = {"used": False}

    with tab_map:
        point = st.session_state.point
        basemap_tile_url = BASEMAP_TEMPLATES["Street view"]
        col_map, col_metrics = st.columns([1.7, 1.3], gap="large")

        rect_size = 0.01
        bounding_box = {
            "north": point["lat"] + rect_size,
            "south": point["lat"] - rect_size,
            "east": point["lon"] + rect_size,
            "west": point["lon"] - rect_size,
        }

        with col_map:
            st.subheader("Map Explorer")

            heatmap_options = [
                "None",
                "Population density",
                "Population total",
                "Tree cover",
                "Water pollution",
                "Land surface temp",
                "Built index",
            ]
            if population_raster_input:
                default_heatmap = "Population density"
            elif population_total_input:
                default_heatmap = "Population total"
            elif treecover_input:
                default_heatmap = "Tree cover"
            elif use_gee:
                default_heatmap = "Water pollution"
            else:
                default_heatmap = "None"

            stored_choice = st.session_state.get("heatmap_choice", default_heatmap)
            if stored_choice not in heatmap_options:
                stored_choice = default_heatmap
            heatmap_choice = st.selectbox(
                "Heatmap overlay",
                heatmap_options,
                index=heatmap_options.index(stored_choice),
                key="heatmap_choice",
            )

            if "heatmap_opacity" not in st.session_state:
                st.session_state["heatmap_opacity"] = 0.75
            heatmap_opacity_input = st.slider(
                "Heatmap opacity",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state["heatmap_opacity"]),
                step=0.05,
                key="heatmap_opacity",
            )

            heatmap_df_map: Optional[pd.DataFrame] = None
            heatmap_label = ""
            heatmap_units = ""
            heatmap_caption = None
            heatmap_error_message = None
            heatmap_color_range: Optional[list[list[int]]] = None
            heatmap_colors: Optional[list[list[int]]] = None
            heatmap_legend: Optional[str] = None
            built_index_median_value: Optional[float] = None

            if heatmap_choice == "Population density":
                if population_raster_input:
                    heatmap_df_raw, heatmap_error_message = raster_dataframe(
                        population_raster_input, MUMBAI_BOUNDS
                    )
                    if heatmap_df_raw is not None:
                        heatmap_colors = HEATMAP_COLOR_SCHEMES["Population density"][
                            "colors"
                        ]
                        heatmap_legend = HEATMAP_COLOR_SCHEMES["Population density"].get(
                            "legend"
                        )
                        heatmap_df_map, vmin, vmax = prepare_heatmap_dataframe(
                            heatmap_df_raw, colors=heatmap_colors
                        )
                        heatmap_df_map["value_display"] = heatmap_df_map["value"].map(
                            lambda v: f"{v:,.0f}"
                        )
                        heatmap_label = "Population density"
                        heatmap_units = "people/km²"
                        heatmap_color_range = heatmap_colors
                        heatmap_caption = (
                            f"{heatmap_label} range ≈ {vmin:,.0f} – {vmax:,.0f} {heatmap_units}"
                        )
                        heatmap_error_message = None
                else:
                    heatmap_error_message = "Population density raster not configured."
            elif heatmap_choice == "Population total":
                if population_total_input:
                    heatmap_df_raw, heatmap_error_message = raster_dataframe(
                        population_total_input, MUMBAI_BOUNDS
                    )
                    if heatmap_df_raw is not None:
                        heatmap_colors = HEATMAP_COLOR_SCHEMES["Population total"][
                            "colors"
                        ]
                        heatmap_legend = HEATMAP_COLOR_SCHEMES["Population total"].get(
                            "legend"
                        )
                        heatmap_df_map, vmin, vmax = prepare_heatmap_dataframe(
                            heatmap_df_raw, colors=heatmap_colors
                        )
                        heatmap_df_map["value_display"] = heatmap_df_map["value"].map(
                            lambda v: f"{v:,.0f}"
                        )
                        heatmap_label = "Population total"
                        heatmap_units = "people"
                        heatmap_color_range = heatmap_colors
                        heatmap_caption = (
                            f"{heatmap_label} range ≈ {vmin:,.0f} – {vmax:,.0f} {heatmap_units}"
                        )
                        heatmap_error_message = None
                else:
                    heatmap_error_message = "Population total raster not configured."
            elif heatmap_choice == "Tree cover":
                if treecover_input:
                    heatmap_df_raw, heatmap_error_message = raster_dataframe(
                        treecover_input, MUMBAI_BOUNDS
                    )
                    if heatmap_df_raw is not None:
                        heatmap_colors = HEATMAP_COLOR_SCHEMES["Tree cover"]["colors"]
                        heatmap_legend = HEATMAP_COLOR_SCHEMES["Tree cover"].get(
                            "legend"
                        )
                        heatmap_df_map, vmin, vmax = prepare_heatmap_dataframe(
                            heatmap_df_raw, colors=heatmap_colors
                        )
                        heatmap_df_map["value_display"] = heatmap_df_map["value"].map(
                            lambda v: f"{v:,.1f}"
                        )
                        heatmap_label = "Tree cover"
                        heatmap_units = "%"
                        heatmap_color_range = heatmap_colors
                        heatmap_caption = (
                            f"{heatmap_label} range ≈ {vmin:,.1f} – {vmax:,.1f} {heatmap_units}"
                        )
                        heatmap_error_message = None
                else:
                    heatmap_error_message = "Tree cover raster not configured."
            elif heatmap_choice == "Water pollution":
                if use_gee and gee_credentials_resolved is not None:
                    heatmap_df_raw, heatmap_error_message = gee_water_pollution_heatmap(
                        MUMBAI_BOUNDS[3],
                        MUMBAI_BOUNDS[1],
                        MUMBAI_BOUNDS[2],
                        MUMBAI_BOUNDS[0],
                        credentials_path=gee_credentials_resolved,
                    )
                    if heatmap_df_raw is not None:
                        heatmap_colors = HEATMAP_COLOR_SCHEMES["Water pollution"]["colors"]
                        heatmap_legend = HEATMAP_COLOR_SCHEMES["Water pollution"].get("legend")
                        heatmap_df_map, vmin, vmax = prepare_heatmap_dataframe(
                            heatmap_df_raw, colors=heatmap_colors
                        )
                        heatmap_df_map["value_display"] = heatmap_df_map["value"].map(
                            lambda v: f"{v:.2f}"
                        )
                        heatmap_label = "Water pollution"
                        heatmap_units = "SWIR ratio"
                        heatmap_color_range = heatmap_colors
                        heatmap_caption = (
                            f"{heatmap_label} index ≈ {vmin:.2f} – {vmax:.2f}"
                        )
                        heatmap_error_message = None
                else:
                    heatmap_error_message = (
                        "Water pollution heatmap requires Earth Engine credentials."
                    )
            elif heatmap_choice == "Land surface temp":
                if use_gee and gee_credentials_resolved is not None:
                    heatmap_df_raw, heatmap_error_message = gee_lst_heatmap(
                        MUMBAI_BOUNDS[3],
                        MUMBAI_BOUNDS[1],
                        MUMBAI_BOUNDS[2],
                        MUMBAI_BOUNDS[0],
                        credentials_path=gee_credentials_resolved,
                    )
                    if heatmap_df_raw is not None:
                        heatmap_colors = HEATMAP_COLOR_SCHEMES["Land surface temp"][
                            "colors"
                        ]
                        heatmap_legend = HEATMAP_COLOR_SCHEMES["Land surface temp"].get(
                            "legend"
                        )
                        heatmap_df_map, vmin, vmax = prepare_heatmap_dataframe(
                            heatmap_df_raw, colors=heatmap_colors
                        )
                        heatmap_df_map["value_display"] = heatmap_df_map["value"].map(
                            lambda v: f"{v:.1f}"
                        )
                        heatmap_label = "Land surface temp"
                        heatmap_units = "°C"
                        heatmap_color_range = heatmap_colors
                        heatmap_caption = (
                            f"{heatmap_label} ≈ {vmin:.1f} – {vmax:.1f} {heatmap_units}"
                        )
                        heatmap_error_message = None
                else:
                    heatmap_error_message = (
                        "Land surface temp heatmap requires Earth Engine credentials."
                    )
            elif heatmap_choice == "Built index":
                if use_gee and gee_credentials_resolved is not None:
                    heatmap_df_raw, heatmap_error_message = gee_built_index_heatmap(
                        MUMBAI_BOUNDS[3],
                        MUMBAI_BOUNDS[1],
                        MUMBAI_BOUNDS[2],
                        MUMBAI_BOUNDS[0],
                        credentials_path=gee_credentials_resolved,
                    )
                    if heatmap_df_raw is not None:
                        valid_values = heatmap_df_raw["value"].dropna()
                        if not valid_values.empty:
                            built_index_median_value = float(valid_values.median())
                            heatmap_df_raw = heatmap_df_raw.copy()
                            heatmap_df_raw["value"] = heatmap_df_raw["value"].fillna(
                                built_index_median_value
                            )
                        heatmap_colors = HEATMAP_COLOR_SCHEMES["Built index"]["colors"]
                        heatmap_legend = HEATMAP_COLOR_SCHEMES["Built index"].get("legend")
                        heatmap_df_map, vmin, vmax = prepare_heatmap_dataframe(
                            heatmap_df_raw, colors=heatmap_colors
                        )
                        heatmap_df_map["value_display"] = heatmap_df_map["value"].map(
                            lambda v: f"{v:.2f}"
                        )
                        heatmap_label = "Built index"
                        heatmap_units = "NDBI"
                        heatmap_color_range = heatmap_colors
                        heatmap_caption = f"{heatmap_label} ≈ {vmin:.2f} – {vmax:.2f}"
                        heatmap_error_message = None
                else:
                    heatmap_error_message = (
                        "Built index heatmap requires Earth Engine credentials."
                    )

            map_deck, _ = map_layers(
                point,
                heatmap_df=heatmap_df_map,
                heatmap_label=heatmap_label,
                heatmap_units=heatmap_units,
                heatmap_color_range=heatmap_color_range,
                basemap_tile_url=basemap_tile_url,
                heatmap_opacity=heatmap_opacity_input,
            )
            selection_state = st.pydeck_chart(
                map_deck,
                selection_mode="single-object",
                on_select="rerun",
                key="map-explorer",
            )
            selected_point = selection_to_point(selection_state)
            if selected_point:
                selected_lat = clamp_value(
                    float(selected_point[0]), MUMBAI_BOUNDS[1], MUMBAI_BOUNDS[3]
                )
                selected_lon = clamp_value(
                    float(selected_point[1]), MUMBAI_BOUNDS[0], MUMBAI_BOUNDS[2]
                )
                current_point = st.session_state.point
                if (
                    abs(current_point["lat"] - selected_lat) > 1e-6
                    or abs(current_point["lon"] - selected_lon) > 1e-6
                ):
                    st.session_state.point = {
                        "lat": selected_lat,
                        "lon": selected_lon,
                    }
                    rerun_fn = getattr(st, "rerun", None) or getattr(
                        st, "experimental_rerun", None
                    )
                    if rerun_fn:
                        rerun_fn()
            st.caption(
                "Tip: Click the map to move the analysis point — the metrics update automatically."
            )
            if heatmap_error_message:
                st.warning(heatmap_error_message)
            elif heatmap_caption:
                st.caption(heatmap_caption)
                if heatmap_legend:
                    legend_html = _heatmap_legend_html(
                        heatmap_label or heatmap_choice,
                        heatmap_colors or [],
                        heatmap_legend,
                    )
                    if legend_html:
                        st.markdown(legend_html, unsafe_allow_html=True)

        population_override = None
        population_total_override = None
        treecover_override = None
        vegetation_override = None
        water_quality_override: Optional[Dict[str, Optional[float]]] = None
        air_quality_override: Optional[Dict[str, Optional[float]]] = None
        lst_override: Optional[float] = None
        lst_source: Optional[str] = None

        if use_gee and gee_credentials_resolved is not None:
            vegetation_status = st.empty()
            with st.spinner("🌱 Analyzing vegetation indices via Earth Engine..."):
                try:
                    vegetation_override = gee_vegetation_indices(
                        bounding_box["north"],
                        bounding_box["south"],
                        bounding_box["east"],
                        bounding_box["west"],
                        credentials_path=gee_credentials_resolved,
                    )
                    vegetation_status.success("✅ Vegetation indices synced from Earth Engine.")
                except EarthEngineUnavailable as exc:
                    vegetation_status.warning(f"⚠️ Earth Engine unavailable: {exc}")
                except Exception as exc:  # pragma: no cover - network dependent
                    vegetation_status.warning(f"⚠️ Vegetation analysis failed: {exc}")

            if vegetation_override and vegetation_override.get("ndbi") is None:
                if built_index_median_value is None:
                    built_df_for_median, _ = gee_built_index_heatmap(
                        bounding_box["north"],
                        bounding_box["south"],
                        bounding_box["east"],
                        bounding_box["west"],
                        credentials_path=gee_credentials_resolved,
                    )
                    if built_df_for_median is not None and not built_df_for_median.empty:
                        valid_values = built_df_for_median["value"].dropna()
                        if not valid_values.empty:
                            built_index_median_value = float(valid_values.median())
                if built_index_median_value is not None:
                    vegetation_override["ndbi"] = built_index_median_value

            water_status = st.empty()
            with st.spinner("🌊 Fetching water-quality proxies via Earth Engine..."):
                try:
                    water_quality_override = gee_water_pollution(
                        bounding_box["north"],
                        bounding_box["south"],
                        bounding_box["east"],
                        bounding_box["west"],
                        credentials_path=gee_credentials_resolved,
                    )
                    water_status.success("✅ Water pollution proxies from Sentinel/Landsat ready.")
                except EarthEngineUnavailable as exc:
                    water_status.warning(f"⚠️ Earth Engine unavailable for water metrics: {exc}")
                except Exception as exc:  # pragma: no cover - network dependent
                    water_status.warning(f"⚠️ Water-quality fetch failed: {exc}")

        air_status = st.empty()
        with st.spinner("🌬️ Gathering local air-quality metrics..."):
            air_quality_override = fetch_air_quality(point["lat"], point["lon"])
        if air_quality_override and any(
            air_quality_override.get(key) is not None for key in ["pm2_5", "no2", "co"]
        ):
            air_status.success("✅ Air-quality indices sourced from Open-Meteo (GEOS-CF).")
        else:
            air_status.warning("⚠️ Air-quality service returned no data; using mock values.")

        lst_status = st.empty()
        if use_gee and gee_credentials_resolved is not None:
            gee_lst_value, gee_lst_message = gee_lst_at_point(
                point["lat"],
                point["lon"],
                credentials_path=gee_credentials_resolved,
            )
            if gee_lst_value is not None:
                lst_override = gee_lst_value
                lst_source = "gee"
                lst_status.success(
                    f"✅ {gee_lst_message or 'Skin temperature via Earth Engine (MODIS).'}"
                )
            else:
                gee_error = gee_lst_message or "Earth Engine returned no land surface temperature."
                nasa_value, nasa_message = fetch_lst(point["lat"], point["lon"])
                if nasa_value is not None:
                    lst_override = nasa_value
                    lst_source = "nasa"
                    fallback_note = (
                        nasa_message or "Skin temperature averaged from NASA POWER (daily TS)."
                    )
                    lst_status.warning(
                        f"⚠️ {gee_error} Falling back to NASA POWER — {fallback_note}"
                    )
                else:
                    fallback_note = nasa_message or "Skin temperature uses mock values."
                    lst_status.warning(f"⚠️ {gee_error} {fallback_note}")
        else:
            nasa_value, nasa_message = fetch_lst(point["lat"], point["lon"])
            if nasa_value is not None:
                lst_override = nasa_value
                lst_source = "nasa"
                lst_status.success(f"✅ {nasa_message}")
            else:
                lst_status.warning(
                    f"⚠️ {nasa_message or 'Skin temperature uses mock values.'}"
                )

        if population_raster_input:
            density_value, density_error = raster_value_from_path(
                population_raster_input, point["lat"], point["lon"]
            )
            if density_error:
                population_status_placeholder.warning(density_error)
            else:
                population_override = density_value
                population_status_placeholder.success(
                    f"Density sample: {population_override:,.0f} /km²"
                )
        if population_total_input:
            pop_total_value, pop_total_error = raster_value_from_path(
                population_total_input, point["lat"], point["lon"]
            )
            if pop_total_error:
                population_total_status_placeholder.warning(pop_total_error)
            else:
                population_total_override = pop_total_value
                population_total_status_placeholder.success(
                    f"Total sample: {population_total_override:,.0f} ppl"
                )
        if treecover_input:
            treecover_value, treecover_error = raster_value_from_path(
                treecover_input, point["lat"], point["lon"]
            )
            if treecover_error:
                treecover_status_placeholder.warning(treecover_error)
            else:
                treecover_override = treecover_value
                treecover_status_placeholder.success(
                    f"Tree cover sample: {treecover_override:,.1f}%"
                )

        region = fetch_region_data(
            point["lat"],
            point["lon"],
            population_density=population_override,
            vegetation_indices=vegetation_override,
            air_quality=air_quality_override,
            water_quality=water_quality_override,
        )
        if lst_override is not None:
            region = replace(region, lst_c=lst_override)
        scores = compute_scores(region)
        recs = recommendations(region, scores)

        vegetation_real = {
            key: bool(vegetation_override and vegetation_override.get(key) is not None)
            for key in ["ndvi", "ndwi", "ndbi", "savi"]
        }
        air_real = {
            key: bool(air_quality_override and air_quality_override.get(key) is not None)
            for key in ["pm2_5", "no2", "co", "co2"]
        }
        population_real = population_override is not None
        population_total_real = population_total_override is not None
        treecover_real = treecover_override is not None
        water_real = bool(water_quality_override and water_quality_override.get("swir_ratio") is not None)

        with col_metrics:
            st.subheader(f"📊 Indicators @ {region.lat:.4f}, {region.lon:.4f}")

            pillar_descriptions = {
                "air": "Blend of PM₂.₅ + NO₂ (higher = cleaner air).",
                "water": "SWIR turbidity proxy (higher = murkier water).",
                "green": "Vegetation cover index (higher = greener).",
                "built": "Normalized difference built-up index (higher = harder surfaces).",
            }

            table_rows = []
            for key in ["air", "water", "green", "built"]:
                table_rows.append(
                    {
                        "Metric": f"{PILLAR_DISPLAY[key]} score",
                        "Value": f"{scores[key]*100:.0f} / 100",
                        "Description": pillar_descriptions[key],
                    }
                )

            raw_metrics = [
                ("PM2.5", f"{region.air_pm25:.1f} µg/m³"),
                ("NO₂ (column)", f"{region.air_no2:.1f} µmol/m²"),
                ("CO₂", f"{region.air_co2:.1f} ppm"),
                ("NDWI", f"{region.ndwi:.2f}"),
                ("NDBI", f"{region.ndbi:.2f}"),
                ("Population density", f"{region.pop_density:,.0f} /km²"),
                ("Population (est.)", f"{population_total_override:,.0f} people" if population_total_override is not None else "—"),
                ("Tree cover", f"{treecover_override:.1f}%" if treecover_override is not None else "—"),
                ("Land surface temp", f"{region.lst_c:.1f} °C" if lst_override is not None else "—"),
            ]

            if not population_real:
                mock_tracker["used"] = True
            if population_total_override is None:
                mock_tracker["used"] = True
            if treecover_override is None:
                mock_tracker["used"] = True
            if lst_override is None:
                mock_tracker["used"] = True

            for label, value in raw_metrics:
                description = METRIC_IMPACTS.get(label, "")
                table_rows.append({"Metric": label, "Value": value, "Description": description})

            st.dataframe(
                pd.DataFrame(table_rows),
                hide_index=True,
                width="stretch",
            )

            st.markdown("### Composite HCI")
            st.metric(label="HCI (0–1, higher better)", value=f"{recs['hci']:.2f}")
            info = METRIC_IMPACTS.get("HCI (0–1, higher better)")
            if info:
                st.caption(info)
            st.plotly_chart(
                _radar_chart(scores),
                config={"responsive": True},
            )

        source_notes = []
        source_notes.append(
            "Air quality sourced from Open-Meteo API." if any(air_real.values()) else "Air quality currently uses mock values."
        )
        source_notes.append(
            "Water layer sampled via Earth Engine SWIR ratios." if water_real else "Water layer currently falls back to mock values."
        )
        source_notes.append(
            "Green cover derived from vegetation indices." if vegetation_real.get("ndvi") else "Green cover currently uses mock vegetation values."
        )
        source_notes.append(
            "Built index derived from Sentinel SWIR/NIR bands." if vegetation_real.get("ndbi") else "Built index currently uses mock values."
        )
        if lst_override is not None:
            if lst_source == "gee":
                source_notes.append("LST (earth skin temp) from MODIS via Earth Engine.")
            elif lst_source == "nasa":
                source_notes.append("LST (earth skin temp) from NASA POWER.")
            else:
                source_notes.append("LST (earth skin temp) updated from remote data.")
        else:
            source_notes.append("LST currently uses mock values.")

        for note in source_notes:
            st.caption(note)

        if (
            not any(air_real.values())
            or not water_real
            or not vegetation_real.get("ndvi")
            or not vegetation_real.get("ndbi")
            or lst_override is None
        ):
            mock_tracker["used"] = True

        scenario_origin_current = (round(region.lat, 5), round(region.lon, 5))
        baseline_ndvi = float(region.ndvi)
        baseline_ndbi = float(region.ndbi)

        if "scenario_origin" not in st.session_state:
            st.session_state["scenario_origin"] = scenario_origin_current
            st.session_state["scenario_ndvi"] = baseline_ndvi
            st.session_state["scenario_ndbi"] = baseline_ndbi
        elif st.session_state.get("scenario_origin") != scenario_origin_current:
            st.session_state["scenario_origin"] = scenario_origin_current
            st.session_state["scenario_ndvi"] = baseline_ndvi
            st.session_state["scenario_ndbi"] = baseline_ndbi

    with tab_ai:
        st.subheader("Scenario simulator")
        st.caption(
            "Adjust greenery and built-up intensity to explore how cooling vegetation or densification"
            " can shift environmental metrics and the overall Healthy City Index."
        )

        control_cols = st.columns([1, 3], gap="large")
        if control_cols[0].button(
            "Reset scenario", help="Revert sliders to current observed values."
        ):
            st.session_state["scenario_origin"] = scenario_origin_current
            st.session_state["scenario_ndvi"] = baseline_ndvi
            st.session_state["scenario_ndbi"] = baseline_ndbi
            rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
            if rerun_fn:
                rerun_fn()

        slider_cols = st.columns(2, gap="large")
        greenery_target = slider_cols[0].slider(
            "Target greenery (NDVI)",
            min_value=0.0,
            max_value=0.9,
            value=float(st.session_state.get("scenario_ndvi", baseline_ndvi)),
            step=0.01,
            help="Higher NDVI represents richer vegetation cover.",
            key="scenario_ndvi",
        )
        built_target = slider_cols[1].slider(
            "Target built index (NDBI)",
            min_value=-0.2,
            max_value=0.7,
            value=float(st.session_state.get("scenario_ndbi", baseline_ndbi)),
            step=0.01,
            help="Higher NDBI signals harder, more impervious surfaces.",
            key="scenario_ndbi",
        )

        scenario_region = simulate_urban_scenario(region, greenery_target, built_target)
        scenario_scores = compute_scores(scenario_region)
        scenario_recs = recommendations(scenario_region, scenario_scores)
        scenario_hci = scenario_recs["hci"]

        st.markdown("### Projected impact")
        impact_cols = st.columns(4, gap="large")
        impact_cols[0].metric(
            "Land surface temp (°C)",
            f"{scenario_region.lst_c:.1f}",
            delta=f"{scenario_region.lst_c - region.lst_c:+.1f} °C",
        )
        impact_cols[1].metric(
            "PM2.5 (µg/m³)",
            f"{scenario_region.air_pm25:.1f}",
            delta=f"{scenario_region.air_pm25 - region.air_pm25:+.1f}",
        )
        impact_cols[2].metric(
            "Water pollution (ratio)",
            f"{scenario_region.water_pollution:.3f}",
            delta=f"{scenario_region.water_pollution - region.water_pollution:+.3f}",
        )
        impact_cols[3].metric(
            "Healthy City Index",
            f"{scenario_hci:.2f}",
            delta=f"{scenario_hci - recs['hci']:+.2f}",
        )

        secondary_cols = st.columns(2, gap="large")
        secondary_cols[0].metric(
            "Green cover (NDVI)",
            f"{scenario_region.ndvi:.3f}",
            delta=f"{scenario_region.ndvi - region.ndvi:+.3f}",
        )
        secondary_cols[1].metric(
            "Built index (NDBI)",
            f"{scenario_region.ndbi:.3f}",
            delta=f"{scenario_region.ndbi - region.ndbi:+.3f}",
        )

        st.markdown("### Scenario guidance")
        st.write("**Habitability:**", scenario_recs["habitability"])
        st.write("**Parks / Greenery:**", scenario_recs["parks"])
        st.write("**Waste Management:**", scenario_recs["waste"])
        st.write("**Disease Risk:**", scenario_recs["disease"])

        st.divider()
        st.subheader("AI Report")
        llm_placeholder = st.empty()
        report_feedback = st.empty()
        download_placeholder = st.empty()

        if st.button("Generate AI Report", type="primary"):
            payload: Dict[str, Any] = {
                "location": {
                    "lat": round(region.lat, 5),
                    "lon": round(region.lon, 5),
                },
                "composite_hci": round(float(scenario_hci), 3),
                "pillar_notes": {
                    "habitability": scenario_recs["habitability"],
                    "parks_greenery": scenario_recs["parks"],
                    "waste_management": scenario_recs["waste"],
                    "disease_risk": scenario_recs["disease"],
                },
                "scores": {key: round(float(value), 3) for key, value in scenario_scores.items()},
                "metrics": {
                    "pm2_5_ugm3": scenario_region.air_pm25,
                    "no2_umolm2": scenario_region.air_no2,
                    "co_umolm2": scenario_region.air_co,
                    "co2_ppm": scenario_region.air_co2,
                    "water_pollution_ratio": scenario_region.water_pollution,
                    "ndvi": scenario_region.ndvi,
                    "ndwi": scenario_region.ndwi,
                    "ndbi": scenario_region.ndbi,
                    "savi": scenario_region.savi,
                    "population_density_per_km2": scenario_region.pop_density,
                    "population_total_estimate": population_total_override,
                    "tree_cover_percent": treecover_override,
                    "land_surface_temp_c": scenario_region.lst_c,
                    "industrial_distance_km": scenario_region.industrial_km,
                    "heatmap_selected": heatmap_choice,
                    "simulated": True,
                },
                "data_quality": {
                    "air": any(air_real.values()),
                    "water": water_real,
                    "green": vegetation_real.get("ndvi", False),
                    "built": vegetation_real.get("ndbi", False),
                },
            }

            try:
                with st.spinner("Consulting urban-planning assistant..."):
                    response_text = generate_plan(payload)
                st.session_state["llm_response"] = response_text
                st.session_state["llm_error"] = None
            except LLMUnavailable as exc:
                st.session_state["llm_response"] = None
                st.session_state["llm_error"] = str(exc)
            except Exception as exc:
                st.session_state["llm_response"] = None
                st.session_state["llm_error"] = f"AI request failed: {exc}"

            ai_text = st.session_state.get("llm_response")
            radar_png = radar_chart_png(
                scenario_scores,
                "Scenario pillar mix",
            )
            map_png = map_snapshot_png(region, scenario_region)
            pdf_bytes = generate_pdf_report(
                region,
                scores,
                recs,
                scenario_region,
                scenario_scores,
                scenario_recs,
                scenario_hci,
                ai_text,
                radar_png=radar_png,
                map_png=map_png,
            )
            st.session_state["ai_report_pdf"] = pdf_bytes
            report_feedback.success("AI report prepared. Download below.")

        if st.session_state.get("llm_response"):
            llm_placeholder.markdown(st.session_state["llm_response"])
        elif st.session_state.get("llm_error"):
            llm_placeholder.warning(st.session_state["llm_error"])
        else:
            llm_placeholder.caption(
                "Tip: export OPENAI_API_KEY before launching the app to unlock AI-generated reports."
            )

        if pdf_bytes := st.session_state.get("ai_report_pdf"):
            download_placeholder.download_button(
                label="Download AI Report (PDF)",
                data=pdf_bytes,
                file_name="healthy-city-report.pdf",
                mime="application/pdf",
            )

    st.divider()
    if mock_tracker["used"]:
        st.caption("* Mock data placeholder while live Earth Engine values are unavailable for the selected area.")
    st.markdown(
        "### Map roadmap"
        "\n1. Replace the blank canvas with cached raster tiles (e.g. Stamen, Carto)"
        " or custom PNGs to avoid live tile requests."
        "\n2. Overlay NASA rasters (LST, NDVI) using `ImageLayer` once processed."
        "\n3. Add vector layers for industrial zones, green spaces, and wards."
        "\n4. Hook the UI to the Python backend endpoint `/api/region?lat=&lon=`."
    )


if __name__ == "__main__":
    main()

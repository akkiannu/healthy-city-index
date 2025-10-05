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
    water_pollution_heatmap as gee_water_pollution_heatmap,
    water_pollution_indices as gee_water_pollution,
)
from hci_app.nasa import fetch_lst, fetch_lst_heatmap
from hci_app.maps import map_layers, prepare_heatmap_dataframe, selection_to_point
from hci_app.models import fetch_region_data
from hci_app.raster import raster_dataframe, raster_value_from_path
from hci_app.scoring import clamp_value, compute_scores, recommendations
from hci_app.llm import LLMUnavailable, generate_plan


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
                heatmap_df_raw, heatmap_info = fetch_lst_heatmap(
                    MUMBAI_BOUNDS[3],
                    MUMBAI_BOUNDS[1],
                    MUMBAI_BOUNDS[2],
                    MUMBAI_BOUNDS[0],
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
                    info_text = heatmap_info or "NASA POWER TS averages"
                    heatmap_caption = (
                        f"{heatmap_label} ≈ {vmin:.1f} – {vmax:.1f} {heatmap_units} · {info_text}"
                    )
                    heatmap_error_message = None
                else:
                    heatmap_error_message = heatmap_info or (
                        "NASA POWER returned no land surface temperature samples."
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
                use_container_width=True,
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
        lst_value, lst_message = fetch_lst(point["lat"], point["lon"])
        if lst_value is not None:
            lst_override = lst_value
            lst_status.success(f"✅ {lst_message}")
        else:
            lst_status.warning(f"⚠️ {lst_message or 'Skin temperature uses mock values.'}")

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
                use_container_width=True,
            )

            st.markdown("### Composite HCI")
            st.metric(label="HCI (0–1, higher better)", value=f"{recs['hci']:.2f}")
            info = METRIC_IMPACTS.get("HCI (0–1, higher better)")
            if info:
                st.caption(info)
            st.plotly_chart(_radar_chart(scores), use_container_width=True)

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
        source_notes.append(
            "LST (earth skin temp) from NASA POWER." if lst_override is not None else "LST currently uses mock values."
        )

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

    with tab_ai:
        col_left, col_right = st.columns([1.2, 1], gap="large")

        with col_left:
            st.subheader("LLM Recommendations (stub)")
            ai_metrics = [
                ("Latitude", f"{region.lat:.5f}", True),
                ("Longitude", f"{region.lon:.5f}", True),
                ("PM2.5", f"{region.air_pm25} μg/m³", air_real.get("pm2_5", False)),
                ("Land surface temp", f"{region.lst_c} °C", False),
                ("Population density", f"{region.pop_density} /km²", population_real),
                ("NO₂ (column)", f"{region.air_no2:.1f} µmol/m²", air_real.get("no2", False)),
                ("CO (column)", f"{region.air_co:.1f} µmol/m²", air_real.get("co", False)),
                ("CO₂", f"{region.air_co2:.1f} ppm", air_real.get("co2", False)),
                ("Water pollution (SWIR ratio)", f"{region.water_pollution:.2f}", water_real),
                ("NDVI", f"{region.ndvi:.3f}", vegetation_real.get("ndvi", False)),
                ("NDWI", f"{region.ndwi:.3f}", vegetation_real.get("ndwi", False)),
                ("NDBI", f"{region.ndbi:.3f}", vegetation_real.get("ndbi", False)),
                ("SAVI", f"{region.savi:.3f}", vegetation_real.get("savi", False)),
            ]
            for label, value, is_real in ai_metrics:
                display_label = f"{label}{'' if is_real else '*'}"
                if not is_real:
                    mock_tracker["used"] = True
                st.metric(display_label, value)
                info = METRIC_IMPACTS.get(label)
                if info:
                    st.caption(info)

        with col_right:
            st.metric("Composite HCI", f"{recs['hci']:.2f}")
            info = METRIC_IMPACTS.get("Composite HCI")
            if info:
                st.caption(info)
            st.write("**Habitability:**", recs["habitability"])
            st.write("**Parks / Greenery:**", recs["parks"])
            st.write("**Waste Management:**", recs["waste"])
            st.write("**Disease Risk:**", recs["disease"])

            st.divider()
            st.subheader("AI Action Plan")
            llm_placeholder = st.empty()

            if st.button("Generate AI plan", type="primary"):
                payload: Dict[str, Any] = {
                    "location": {
                        "lat": round(region.lat, 5),
                        "lon": round(region.lon, 5),
                    },
                    "composite_hci": round(float(recs["hci"]), 3),
                    "pillar_notes": {
                        "habitability": recs["habitability"],
                        "parks_greenery": recs["parks"],
                        "waste_management": recs["waste"],
                        "disease_risk": recs["disease"],
                    },
                    "scores": {key: round(float(value), 3) for key, value in scores.items()},
                    "metrics": {
                        "pm2_5_ugm3": region.air_pm25,
                        "no2_umolm2": region.air_no2,
                        "co_umolm2": region.air_co,
                        "co2_ppm": region.air_co2,
                        "water_pollution_ratio": region.water_pollution,
                        "ndvi": region.ndvi,
                        "ndwi": region.ndwi,
                        "ndbi": region.ndbi,
                        "savi": region.savi,
                        "population_density_per_km2": region.pop_density,
                        "population_total_estimate": population_total_override,
                        "tree_cover_percent": treecover_override,
                        "land_surface_temp_c": region.lst_c,
                        "industrial_distance_km": region.industrial_km,
                        "heatmap_selected": heatmap_choice,
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

            if st.session_state.get("llm_response"):
                llm_placeholder.markdown(st.session_state["llm_response"])
            elif st.session_state.get("llm_error"):
                llm_placeholder.warning(st.session_state["llm_error"])
            else:
                llm_placeholder.caption(
                    "Tip: export OPENAI_API_KEY before launching the app to unlock AI-generated action plans."
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

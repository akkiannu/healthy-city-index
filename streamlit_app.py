"""Streamlit version of the Healthy City Index wireframe.

This script mirrors the existing React prototype, providing a Python-first
implementation with mocked indicator data, scoring, and recommendations. The
map section uses pydeck with a blank base layer so the app remains functional
without pulling remote tiles. Future iterations can swap in cached rasters or
custom vector layers once NASA data products are ready.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

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
)
from hci_app.airquality import fetch_air_quality
from hci_app.earthengine import (
    DEFAULT_CREDENTIALS_PATH,
    EarthEngineUnavailable,
    vegetation_indices as gee_vegetation_indices,
    water_pollution_heatmap as gee_water_pollution_heatmap,
    water_pollution_indices as gee_water_pollution,
)
from hci_app.maps import map_layers, prepare_heatmap_dataframe, selection_to_point
from hci_app.models import fetch_region_data
from hci_app.raster import raster_dataframe, raster_value_from_path
from hci_app.scoring import clamp_value, compute_scores, recommendations
from hci_app.llm import LLMUnavailable, llm_recommendations


st.set_page_config(page_title="Healthy City Index — Mumbai", layout="wide")


METRIC_IMPACTS: Dict[str, str] = {
    "PM2.5": "Fine particulates drive respiratory risk and spotlight emission-control needs.",
    "NO₂ (column)": "Reactive nitrogen traces traffic pressure and street ventilation gaps.",
    "CO (column)": "Carbon monoxide reveals incomplete combustion hotspots for transit or fuel fixes.",
    "CO₂": "Local carbon load signals where efficiency retrofits and clean power matter most.",
    "Water pollution (SWIR ratio)": "High ratios hint at turbid or polluted waters needing remediation.",
    "NDVI": "Vegetation coverage guides cooling corridors and nature-based infrastructure sites.",
    "NDWI": "Surface moisture marks wetlands or flood buffers to safeguard.",
    "NDBI": "Built-up intensity shows hardscape concentration for balancing land use.",
    "SAVI": "Vegetation health flags parks needing irrigation or soil rehabilitation.",
    "Population density": "Where residents cluster, services and evacuation plans must scale.",
    "Population (est.)": "Total headcount informs school, health, and mobility capacity planning.",
    "Tree cover": "Tree canopy cools streetscapes and improves air—priority zones for preservation.",
    "Land surface temp": "Hot surfaces expose heat islands requiring shade or cool-roof programs.",
    "Industrial proximity": "Nearby industry demands buffers and compatibility checks in zoning.",
    "HCI (0–1, higher better)": "Composite livability score to target the weakest planning dimension.",
    "Composite HCI": "Narrative indicator for briefing leadership on overall urban resilience.",
    "Latitude": "Anchor coordinate to align zoning and infrastructure overlays.",
    "Longitude": "Works with latitude to place interventions within cadastral grids.",
    "LST": "Thermal readings pinpoint microclimate stress needing cooling interventions.",
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
    categories = ["Air", "Water", "Green", "Population", "Temperature", "Industrial"]
    values = [round(scores[key.lower()], 3) * 100 for key in categories]
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

    low_label = "Low"
    high_label = "High"
    if legend_text:
        parts = [part.strip() for part in legend_text.split("·")]
        if parts:
            low_label = parts[0]
            if len(parts) > 1:
                high_label = parts[1]

    gradient = ", ".join(stops)
    return f"""
<style>
.hci-legend {{
  display: flex;
  align-items: center;
  font-size: 0.7rem;
  gap: 0.75rem;
  margin-top: 0.35rem;
  margin-bottom: 0.5rem;
}}
.hci-legend-name {{
  font-weight: 600;
}}
.hci-legend-bar {{
  flex: 1 1 auto;
  height: 12px;
  border-radius: 999px;
  background: linear-gradient(90deg, {gradient});
  border: 1px solid rgba(15, 23, 42, 0.08);
}}
.hci-legend-text {{
  line-height: 1;
}}
</style>
<div class="hci-legend">
  <span class="hci-legend-name">{name}</span>
  <div class="hci-legend-bar"></div>
  <span class="hci-legend-text">{low_label}</span>
  <span class="hci-legend-text">{high_label}</span>
</div>
"""


def main() -> None:
    _initialise_session_state()

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
        population_status_placeholder.info("Population density currently uses mock values.")
    if not population_total_input:
        population_total_status_placeholder.info("Population totals are not configured.")
    if not treecover_input:
        treecover_status_placeholder.info("Tree cover overlay not configured.")

    st.sidebar.header("Map appearance")
    default_style = "Street view"
    basemap_options = list(BASEMAP_TEMPLATES.keys())
    st.session_state.setdefault("basemap_style", default_style)
    current_style = st.session_state.get("basemap_style", default_style)
    basemap_style = st.sidebar.selectbox(
        "Base map",
        basemap_options,
        index=basemap_options.index(current_style) if current_style in basemap_options else 0,
        help="Switch between street, satellite, or terrain tiles.",
    )
    st.session_state["basemap_style"] = basemap_style
    basemap_tile_url = BASEMAP_TEMPLATES[basemap_style]

    st.sidebar.subheader("Heatmap controls")
    st.session_state.setdefault("heatmap_opacity", 0.75)
    heatmap_opacity_input = st.sidebar.slider(
        "Heatmap opacity",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["heatmap_opacity"]),
        step=0.05,
    )
    st.session_state["heatmap_opacity"] = heatmap_opacity_input

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

    heatmap_options = [
        "None",
        "Population density",
        "Population total",
        "Tree cover",
        "Water pollution",
    ]
    default_heatmap_index = 0
    if population_raster_input:
        default_heatmap_index = 1
    elif treecover_input:
        default_heatmap_index = 3
    heatmap_choice = st.sidebar.selectbox(
        "Heatmap overlay",
        heatmap_options,
        index=default_heatmap_index,
        help="Choose a layer to visualise as a heat map across Mumbai.",
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
        col_map, col_metrics = st.columns([1.7, 1.3], gap="large")

        rect_size = 0.01
        bounding_box = {
            "north": point["lat"] + rect_size,
            "south": point["lat"] - rect_size,
            "east": point["lon"] + rect_size,
            "west": point["lon"] - rect_size,
        }

        with col_map:
            st.subheader("Pick a location")
            with st.form(key="location_form", clear_on_submit=False):
                lat_value = st.number_input(
                    "Latitude",
                    min_value=18.5,
                    max_value=20.0,
                    value=float(st.session_state.point["lat"]),
                    step=0.0005,
                    format="%.4f",
                )
                lon_value = st.number_input(
                    "Longitude",
                    min_value=72.5,
                    max_value=73.5,
                    value=float(st.session_state.point["lon"]),
                    step=0.0005,
                    format="%.4f",
                )
                submitted = st.form_submit_button("Update location")

            if submitted:
                st.session_state.point = {"lat": lat_value, "lon": lon_value}
                point = st.session_state.point
            else:
                point = st.session_state.point

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

        if population_raster_input:
            density_value, density_error = raster_value_from_path(
                population_raster_input, point["lat"], point["lon"]
            )
            if density_error:
                population_status_placeholder.warning(density_error)
            else:
                population_override = density_value
                population_status_placeholder.success(
                    f"Population density sample: {population_override:,.0f} people/km²"
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
                    f"Population total sample: {population_total_override:,.0f} people"
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
            st.subheader(
                f"📊 Indicators @ {region.lat:.4f}, {region.lon:.4f}"
            )

            metrics = [
                ("PM2.5", f"{region.air_pm25:.1f} µg/m³", air_real.get("pm2_5", False)),
                ("NO₂ (column)", f"{region.air_no2:.1f} µmol/m²", air_real.get("no2", False)),
                ("CO (column)", f"{region.air_co:.1f} µmol/m²", air_real.get("co", False)),
                ("CO₂", f"{region.air_co2:.1f} ppm", air_real.get("co2", False)),
                ("Water pollution (SWIR ratio)", f"{region.water_pollution:.2f}", water_real),
                ("NDVI", f"{region.ndvi:.2f}", vegetation_real.get("ndvi", False)),
                ("NDWI", f"{region.ndwi:.2f}", vegetation_real.get("ndwi", False)),
                ("NDBI", f"{region.ndbi:.2f}", vegetation_real.get("ndbi", False)),
                ("SAVI", f"{region.savi:.2f}", vegetation_real.get("savi", False)),
                ("Population density", f"{region.pop_density:,.0f} /km²", population_real),
                ("Land surface temp", f"{region.lst_c:.1f} °C", False),
                ("Industrial proximity", f"{region.industrial_km:.2f} km", False),
            ]

            for entry in metrics:
                if len(entry) == 3:
                    label, value, is_real = entry
                else:
                    label, value = entry
                    is_real = True
                display_label = f"{label}{'' if is_real else '*'}"
                if not is_real:
                    mock_tracker["used"] = True
                st.metric(label=display_label, value=value)
                info = METRIC_IMPACTS.get(label)
                if info:
                    st.caption(info)

            if population_total_override is not None:
                st.metric("Population (est.)", f"{population_total_override:,.0f} people")
                info = METRIC_IMPACTS.get("Population (est.)")
                if info:
                    st.caption(info)
            if treecover_override is not None:
                st.metric("Tree cover", f"{treecover_override:,.1f}%")
                info = METRIC_IMPACTS.get("Tree cover")
                if info:
                    st.caption(info)

            st.markdown("### Composite HCI")
            st.metric(label="HCI (0–1, higher better)", value=f"{recs['hci']:.2f}")
            info = METRIC_IMPACTS.get("HCI (0–1, higher better)")
            if info:
                st.caption(info)
            st.plotly_chart(_radar_chart(scores), use_container_width=True)

            if (
                population_override is not None
                and population_total_override is not None
                and treecover_override is not None
            ):
                st.caption(
                    "Population density, population totals, and tree cover are sourced"
                    " from the provided rasters. Other indicators remain mocked until"
                    " their data sources are wired."
                )
            elif population_override is not None and population_total_override is not None:
                st.caption(
                    "Population density and total population sourced from the"
                    " provided rasters. Other indicators remain mocked until"
                    " their data sources are wired."
                )
            elif population_override is not None and treecover_override is not None:
                st.caption(
                    "Population density and tree cover sourced from the provided"
                    " rasters. Remaining indicators still use mock values."
                )
            elif population_override is not None:
                st.caption(
                    "Population density sourced from the provided raster. Other"
                    " indicators remain mocked until their data sources are wired."
                )
            elif population_total_override is not None:
                st.caption(
                    "Population totals sourced from the provided raster. Density"
                    " and the remaining indicators still use mock values."
                )
            elif treecover_override is not None:
                st.caption(
                    "Tree cover values sourced from the provided raster. All other"
                    " indicators currently use mock values."
                )
            else:
                st.caption(
                    "⚠️ Mock values for wireframe. Swap `fetch_region_data` with"
                    " backend calls as data sources come online."
                )

    with tab_ai:
        col_left, col_right = st.columns([1.2, 1], gap="large")

        with col_left:
            st.subheader("LLM Recommendations (stub)")
            ai_metrics = [
                ("Latitude", f"{region.lat:.5f}", True),
                ("Longitude", f"{region.lon:.5f}", True),
                ("PM2.5", f"{region.air_pm25} μg/m³", air_real.get("pm2_5", False)),
                ("LST", f"{region.lst_c} °C", False),
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
            st.caption(
                "Swap this panel with real LLM outputs by forwarding indicators and"
                " scores as structured context."
            )

            st.divider()
            st.subheader("AI action plan")
            llm_placeholder = st.empty()
            st.session_state.setdefault("llm_response", None)

            def _llm_payload() -> Dict:
                return {
                    "location": {
                        "lat": round(region.lat, 5),
                        "lon": round(region.lon, 5),
                    },
                    "scores": {k: float(v) for k, v in scores.items()},
                    "composite_hci": float(recs["hci"]),
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
                        "land_surface_temp_c": region.lst_c,
                        "industrial_distance_km": region.industrial_km,
                    },
                    "data_sources": {
                        "population_density": "raster" if population_real else "mock",
                        "population_total": "raster" if population_total_real else "mock",
                        "tree_cover": "raster" if treecover_real else "mock",
                        "air_quality": "open-meteo API" if any(air_real.values()) else "mock",
                        "water_pollution": "earth engine" if water_real else "mock",
                    },
                    "heatmap": heatmap_choice,
                }

            if st.button("Generate AI plan", type="primary"):
                try:
                    with st.spinner("Consulting urban-planning assistant..."):
                        payload = _llm_payload()
                        response_text = llm_recommendations(payload)
                        st.session_state["llm_response"] = response_text
                except LLMUnavailable as exc:
                    st.session_state["llm_response"] = None
                    llm_placeholder.warning(str(exc))
                except Exception as exc:  # pragma: no cover
                    st.session_state["llm_response"] = None
                    llm_placeholder.error(f"AI request failed: {exc}")

            if st.session_state["llm_response"]:
                llm_placeholder.markdown(st.session_state["llm_response"])
            elif not st.session_state.get("llm_response"):
                llm_placeholder.caption(
                    "Tip: set OPENAI_API_KEY in your environment to enable AI-generated action plans."
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

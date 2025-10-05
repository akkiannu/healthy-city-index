"""Streamlit version of the Healthy City Index wireframe.

This script mirrors the existing React prototype, providing a Python-first
implementation with mocked indicator data, scoring, and recommendations. The
map section uses pydeck with a blank base layer so the app remains functional
without pulling remote tiles. Future iterations can swap in cached rasters or
custom vector layers once NASA data products are ready.
"""

from __future__ import annotations

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
from hci_app.maps import map_layers, prepare_heatmap_dataframe, selection_to_point
from hci_app.models import fetch_region_data
from hci_app.raster import raster_dataframe, raster_value_from_path
from hci_app.scoring import clamp_value, compute_scores, recommendations


st.set_page_config(page_title="Healthy City Index — Mumbai", layout="wide")


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

    heatmap_options = ["None", "Population density", "Population total", "Tree cover"]
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

    with tab_map:
        point = st.session_state.point
        col_map, col_metrics = st.columns([1.7, 1.3], gap="large")

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

            if heatmap_choice == "Population density":
                if population_raster_input:
                    heatmap_df_raw, heatmap_error_message = raster_dataframe(
                        population_raster_input, MUMBAI_BOUNDS
                    )
                    if heatmap_df_raw is not None:
                        heatmap_colors = HEATMAP_COLOR_SCHEMES["Population density"][
                            "colors"
                        ]
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

            map_deck, _ = map_layers(
                point,
                heatmap_df=heatmap_df_map,
                heatmap_label=heatmap_label,
                heatmap_units=heatmap_units,
                heatmap_color_range=heatmap_color_range,
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
            st.info(
                "This draft uses a blank pydeck canvas. Next iterations can add cached"
                " tiles or NASA vector overlays once data is prepared.",
                icon="ℹ️",
            )
            st.caption(
                "Tip: Click the map to move the analysis point — the metrics update automatically."
            )
            if heatmap_error_message:
                st.warning(heatmap_error_message)
            elif heatmap_caption:
                st.caption(heatmap_caption)

        population_override = None
        population_total_override = None
        treecover_override = None

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
            point["lat"], point["lon"], population_density=population_override
        )
        scores = compute_scores(region)
        recs = recommendations(region, scores)

        with col_metrics:
            st.subheader(
                f"📊 Indicators @ {region.lat:.4f}, {region.lon:.4f}"
            )

            metrics = [
                ("PM2.5", f"{region.air_pm25:.1f} μg/m³"),
                ("NO₂", f"{region.air_no2:.1f} μg/m³"),
                ("Water turbidity", f"{region.water_turbidity:.1f} NTU"),
                ("NDVI", f"{region.ndvi:.2f}"),
                ("Population density", f"{region.pop_density:,.0f} /km²"),
                ("Land surface temp", f"{region.lst_c:.1f} °C"),
                ("Industrial proximity", f"{region.industrial_km:.2f} km"),
            ]

            for label, value in metrics:
                st.metric(label=label, value=value)

            if population_total_override is not None:
                st.metric("Population (est.)", f"{population_total_override:,.0f} people")
            if treecover_override is not None:
                st.metric("Tree cover", f"{treecover_override:,.1f}%")

            st.markdown("### Composite HCI")
            st.metric(label="HCI (0–1, higher better)", value=f"{recs['hci']:.2f}")
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
            st.metric("Latitude", f"{region.lat:.5f}")
            st.metric("Longitude", f"{region.lon:.5f}")
            st.metric("PM2.5", f"{region.air_pm25} μg/m³")
            st.metric("LST", f"{region.lst_c} °C")
            st.metric("Population density", f"{region.pop_density} /km²")
            st.metric("Water turbidity", f"{region.water_turbidity} NTU")

        with col_right:
            st.metric("Composite HCI", f"{recs['hci']:.2f}")
            st.write("**Habitability:**", recs["habitability"])
            st.write("**Parks / Greenery:**", recs["parks"])
            st.write("**Waste Management:**", recs["waste"])
            st.write("**Disease Risk:**", recs["disease"])
            st.caption(
                "Swap this panel with real LLM outputs by forwarding indicators and"
                " scores as structured context."
            )

    st.divider()
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

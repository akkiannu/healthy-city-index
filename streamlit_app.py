"""Streamlit version of the Healthy City Index wireframe.

This script mirrors the existing React prototype, providing a Python-first
implementation with mocked indicator data, scoring, and recommendations. The
map section uses pydeck with a blank base layer so the app remains functional
without pulling remote tiles. Future iterations can swap in cached rasters or
custom vector layers once NASA data products are ready.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st


st.set_page_config(page_title="Healthy City Index — Mumbai", layout="wide")


@dataclass(frozen=True)
class RegionData:
    lat: float
    lon: float
    air_pm25: float
    air_no2: float
    water_turbidity: float
    ndvi: float
    pop_density: float
    lst_c: float
    industrial_km: float


DEFAULT_POINT: Dict[str, float] = {"lat": 19.0760, "lon": 72.8777}
DEFAULT_REGION = RegionData(
    lat=DEFAULT_POINT["lat"],
    lon=DEFAULT_POINT["lon"],
    air_pm25=40.0,
    air_no2=25.0,
    water_turbidity=5.0,
    ndvi=0.40,
    pop_density=12_000,
    lst_c=32.0,
    industrial_km=3.0,
)


def _noise(lat: float, lon: float, seed: int) -> float:
    return abs(math.sin(lat * 5 + lon * 3 + seed)) % 1.0


def fetch_region_data(lat: float, lon: float) -> RegionData:
    n = [_noise(lat, lon, i) for i in range(1, 8)]
    try:
        return RegionData(
            lat=lat,
            lon=lon,
            air_pm25=round(20 + 80 * n[0], 1),
            air_no2=round(10 + 60 * n[1], 1),
            water_turbidity=round(1 + 15 * n[2], 1),
            ndvi=round(0.15 + 0.6 * n[3], 2),
            pop_density=int(round(3000 + 25_000 * n[4])),
            lst_c=round(28 + 10 * n[5], 1),
            industrial_km=round(0.1 + 8.0 * n[6], 2),
        )
    except Exception:
        return DEFAULT_REGION


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _minmax(v: float, lo: float, hi: float, invert: bool = False) -> float:
    x = _clamp(v, lo, hi)
    score = (x - lo) / (hi - lo or 1)
    return 1 - score if invert else score


def compute_scores(data: RegionData) -> Dict[str, float]:
    return {
        "air": 0.5 * _minmax(data.air_pm25, 10, 100, invert=True)
        + 0.5 * _minmax(data.air_no2, 5, 80, invert=True),
        "water": _minmax(data.water_turbidity, 1, 20, invert=True),
        "green": _minmax(data.ndvi, 0.1, 0.8),
        "population": _minmax(data.pop_density, 1_000, 30_000, invert=True),
        "temperature": _minmax(data.lst_c, 26, 40, invert=True),
        "industrial": _minmax(data.industrial_km, 0.1, 10.0),
    }


def composite(scores: Dict[str, float]) -> float:
    weights = {
        "air": 0.22,
        "water": 0.14,
        "green": 0.18,
        "population": 0.12,
        "temperature": 0.20,
        "industrial": 0.14,
    }
    value = sum(scores[k] * weights[k] for k in weights)
    return round(value, 3)


def recommendations(data: RegionData, scores: Dict[str, float]) -> Dict[str, str | float]:
    hci = composite(scores)
    habitability = (
        "Generally habitable – favorable profile."
        if hci >= 0.7
        else "Marginally habitable – mixed; targeted fixes."
        if hci >= 0.5
        else "Not ideal – multiple risks; mitigate first."
    )

    if data.ndvi >= 0.45:
        parks = "Adequate greenery; preserve & add pocket parks."
    elif data.ndvi >= 0.30:
        parks = "Moderate greenery; corridor greening & shade trees."
    else:
        parks = "Low greenery; prioritize parks & streetscape planting."

    if data.pop_density > 15_000 and data.industrial_km < 2:
        waste = "High waste pressure; deploy MRF/transfer stations & audits."
    elif data.pop_density > 15_000:
        waste = "Elevated waste; scale collection & segregation."
    else:
        waste = "Standard services likely sufficient; maintain programs."

    risk = sum(
        [
            data.air_pm25 > 60,
            data.lst_c > 34,
            data.pop_density > 20_000,
            data.water_turbidity > 10,
        ]
    )
    disease = (
        "High risk: clinics, heat shelters, vector control, potable water."
        if risk >= 3
        else "Moderate risk: monitor hotspots, shade/water points, seasonal drives."
        if risk == 2
        else "Low–moderate risk: routine surveillance & outreach."
    )

    return {
        "hci": hci,
        "habitability": habitability,
        "parks": parks,
        "waste": waste,
        "disease": disease,
    }


def _initialise_session_state() -> None:
    if "point" not in st.session_state:
        st.session_state.point = DEFAULT_POINT.copy()


def _map_layers(point: Dict[str, float]) -> Tuple[pdk.Deck, Dict[str, float]]:
    view_state = pdk.ViewState(
        latitude=point["lat"], longitude=point["lon"], zoom=11, pitch=0
    )

    # Industrial proxy rectangle around central Mumbai (mock overlay).
    rectangle = [
        [72.82, 19.02],
        [72.93, 19.02],
        [72.93, 19.10],
        [72.82, 19.10],
    ]

    rectangle_layer = pdk.Layer(
        "PolygonLayer",
        data=[{"polygon": rectangle}],
        get_polygon="polygon",
        get_fill_color=[252, 165, 165, 60],
        get_line_color=[248, 113, 113, 200],
        line_width_min_pixels=1,
    )

    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame([point]),
        get_position="[lon, lat]",
        get_radius=180,
        get_fill_color=[30, 64, 175, 200],
        get_line_color=[15, 23, 42, 240],
        line_width_min_pixels=1,
    )

    deck = pdk.Deck(
        map_style=None,
        initial_view_state=view_state,
        layers=[rectangle_layer, point_layer],
        tooltip={"text": "Lat: {lat}\nLon: {lon}"},
    )
    return deck, view_state.__dict__


def _radar_chart(scores: Dict[str, float]) -> go.Figure:
    categories = ["Air", "Water", "Green", "Population", "Temperature", "Industrial"]
    values = [round(scores[k.lower()], 3) * 100 for k in categories]
    return go.Figure(
        data=go.Scatterpolar(r=values + values[:1], theta=categories + categories[:1], fill="toself")
    ).update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )


def main() -> None:
    _initialise_session_state()

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

            map_deck, _ = _map_layers(st.session_state.point)
            st.pydeck_chart(map_deck, use_container_width=True)
            st.info(
                "This draft uses a blank pydeck canvas. Next iterations can add cached"
                " tiles or NASA vector overlays once data is prepared.",
                icon="ℹ️",
            )

        point = st.session_state.point
        region = fetch_region_data(point["lat"], point["lon"])
        scores = compute_scores(region)
        recs = recommendations(region, scores)

        with col_metrics:
            st.subheader(
                f"📊 Indicators @ {region.lat:.4f}, {region.lon:.4f}"
            )

            metrics = [
                ("PM2.5", f"{region.air_pm25} μg/m³"),
                ("NO₂", f"{region.air_no2} μg/m³"),
                ("Water turbidity", f"{region.water_turbidity} NTU"),
                ("NDVI", f"{region.ndvi}"),
                ("Population density", f"{region.pop_density} /km²"),
                ("Land surface temp", f"{region.lst_c} °C"),
                ("Industrial proximity", f"{region.industrial_km} km"),
            ]

            for label, value in metrics:
                st.metric(label=label, value=value)

            st.markdown("### Composite HCI")
            st.metric(label="HCI (0–1, higher better)", value=f"{recs['hci']:.2f}")
            st.plotly_chart(_radar_chart(scores), use_container_width=True)

            st.caption(
                "⚠️ Mock values for wireframe. Swap `fetch_region_data` with backend"
                " calls when the Python API is ready."
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

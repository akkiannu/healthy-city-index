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
import folium
from streamlit_folium import st_folium
from src.greenery.vegetation import analyze_vegetation


st.set_page_config(page_title="Healthy City Index — Mumbai", layout="wide")


@dataclass(frozen=True)
class RegionData:
    lat: float
    lon: float
    ndvi: float
    ndwi: float
    ndbi: float
    savi: float


DEFAULT_POINT: Dict[str, float] = {"lat": 19.0760, "lon": 72.8777}
DEFAULT_REGION = RegionData(
    lat=DEFAULT_POINT["lat"],
    lon=DEFAULT_POINT["lon"],
    ndvi=0.0,
    ndwi=0.0,
    ndbi=0.0,
    savi=0.0,
)


def fetch_region_data(lat: float, lon: float, bounding_box: dict = None) -> RegionData:
    # Initialize vegetation data with default values
    vegetation_data = {
        "ndvi": 0.0,
        "ndwi": 0.0,
        "ndbi": 0.0,
        "savi": 0.0,
    }
    
    # Get real vegetation data if bounding box is provided
    if bounding_box:
        try:
            st.info("🌱 Analyzing vegetation data... This may take a few seconds.")
            vegetation_data = analyze_vegetation(
                bounding_box["north"], 
                bounding_box["south"], 
                bounding_box["east"], 
                bounding_box["west"]
            )
            st.success("✅ Vegetation analysis completed!")
        except Exception as e:
            st.error(f"❌ Vegetation analysis failed: {e}")
            return DEFAULT_REGION
    
    try:
        return RegionData(
            lat=lat,
            lon=lon,
            ndvi=vegetation_data["ndvi"],
            ndwi=vegetation_data["ndwi"],
            ndbi=vegetation_data["ndbi"],
            savi=vegetation_data["savi"],
        )
    except Exception as e:
        st.error(f"Error creating RegionData: {e}")
        return DEFAULT_REGION


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _minmax(v: float, lo: float, hi: float, invert: bool = False) -> float:
    x = _clamp(v, lo, hi)
    score = (x - lo) / (hi - lo or 1)
    return 1 - score if invert else score


def compute_scores(data: RegionData) -> Dict[str, float]:
    return {
        "ndvi": _minmax(data.ndvi, 0.1, 0.8),
        "ndwi": _minmax(data.ndwi, 0.1, 0.8),
        "ndbi": _minmax(data.ndbi, 0.1, 0.8),
        "savi": _minmax(data.savi, 0.1, 0.8),
    }


def composite(scores: Dict[str, float]) -> float:
    weights = {
        "ndvi": 0.4,
        "ndwi": 0.2,
        "ndbi": 0.2,
        "savi": 0.2,
    }
    value = sum(scores[k] * weights[k] for k in weights)
    return round(value, 3)


def recommendations(data: RegionData, scores: Dict[str, float]) -> Dict[str, str | float]:
    hci = composite(scores)
    habitability = (
        "Excellent vegetation health – very green area."
        if hci >= 0.7
        else "Good vegetation health – moderate greenery."
        if hci >= 0.5
        else "Poor vegetation health – needs improvement."
    )

    if data.ndvi >= 0.45:
        parks = "High vegetation density; preserve existing greenery."
    elif data.ndvi >= 0.30:
        parks = "Moderate vegetation; consider adding more green spaces."
    else:
        parks = "Low vegetation; prioritize planting trees and creating parks."

    if data.ndwi >= 0.3:
        water = "Good water presence; maintain water bodies."
    elif data.ndwi >= 0.1:
        water = "Moderate water presence; consider water conservation."
    else:
        water = "Low water presence; implement water management strategies."

    if data.ndbi >= 0.2:
        builtup = "High built-up area; balance with green infrastructure."
    elif data.ndbi >= 0.1:
        builtup = "Moderate built-up area; maintain green balance."
    else:
        builtup = "Low built-up area; good for natural development."

    return {
        "hci": hci,
        "habitability": habitability,
        "parks": parks,
        "water": water,
        "builtup": builtup,
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
    categories = ["NDVI", "NDWI", "NDBI", "SAVI"]
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

    st.title("🌱 Vegetation Analysis — Mumbai")
    st.caption("Real-time satellite data analysis using Google Earth Engine")

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
            
            # Manual coordinate input
            with st.form(key="location_form", clear_on_submit=False):
                col_lat, col_lon = st.columns(2)
                with col_lat:
                    lat_value = st.number_input(
                        "Latitude",
                        min_value=18.5,
                        max_value=20.0,
                        value=float(st.session_state.point["lat"]),
                        step=0.0005,
                        format="%.4f",
                    )
                with col_lon:
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

            # Interactive map with click functionality using Folium
            st.markdown("**🗺️ Click anywhere on the map to update location:**")
            
            # Create a Folium map
            m = folium.Map(
                location=[st.session_state.point["lat"], st.session_state.point["lon"]], 
                zoom_start=11,
                tiles='OpenStreetMap'
            )
            
            # Add a marker for the current location
            folium.Marker(
                [st.session_state.point["lat"], st.session_state.point["lon"]],
                popup=f"Current Location<br>Lat: {st.session_state.point['lat']:.4f}<br>Lon: {st.session_state.point['lon']:.4f}",
                tooltip="Current Location",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)
            
            # Add dynamic industrial area overlay based on current location
            # Create a rectangle around the current point (simulating industrial proximity)
            current_lat = st.session_state.point["lat"]
            current_lon = st.session_state.point["lon"]
            
            # Create a small rectangle around the current location
            rect_size = 0.01  # Adjust this to make the rectangle bigger/smaller
            industrial_bounds = [
                [current_lat - rect_size, current_lon - rect_size],
                [current_lat + rect_size, current_lon + rect_size]
            ]
            
            folium.Rectangle(
                bounds=industrial_bounds,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.2,
            ).add_to(m)
            
            # Display the map and capture click events
            map_data = st_folium(m, height=400, width=700, returned_objects=["last_clicked"])
            
            # Handle map clicks
            if map_data and map_data.get("last_clicked"):
                clicked_lat = map_data["last_clicked"]["lat"]
                clicked_lon = map_data["last_clicked"]["lng"]
                
                # Update session state if coordinates changed significantly
                if (abs(clicked_lat - st.session_state.point["lat"]) > 0.0001 or 
                    abs(clicked_lon - st.session_state.point["lon"]) > 0.0001):
                    st.session_state.point = {"lat": clicked_lat, "lon": clicked_lon}
                    st.success(f"📍 Location updated to: {clicked_lat:.4f}, {clicked_lon:.4f}")
                    st.rerun()
            
            st.info(
                "Click anywhere on the map above to update the location. The blue marker shows your current selection.",
                icon="ℹ️",
            )

        point = st.session_state.point
        
        # Create bounding box for vegetation analysis
        rect_size = 0.01  # Same size as the rectangle on the map
        bounding_box = {
            "north": point["lat"] + rect_size,
            "south": point["lat"] - rect_size,
            "east": point["lon"] + rect_size,
            "west": point["lon"] - rect_size
        }
        
        region = fetch_region_data(point["lat"], point["lon"], bounding_box)
        scores = compute_scores(region)
        recs = recommendations(region, scores)

        with col_metrics:
            st.subheader(
                f"📊 Vegetation Indices @ {region.lat:.4f}, {region.lon:.4f}"
            )

            metrics = [
                ("NDVI", f"{region.ndvi:.4f}"),
                ("NDWI", f"{region.ndwi:.4f}"),
                ("NDBI", f"{region.ndbi:.4f}"),
                ("SAVI", f"{region.savi:.4f}"),
            ]

            for label, value in metrics:
                st.metric(label=label, value=value)

            st.markdown("### Vegetation Health Score")
            st.metric(label="VHS (0–1, higher better)", value=f"{recs['hci']:.2f}")
            st.plotly_chart(_radar_chart(scores), use_container_width=True)

            st.caption(
                "🌱 All vegetation indices are calculated from real satellite data using Google Earth Engine!"
            )

    with tab_ai:
        col_left, col_right = st.columns([1.2, 1], gap="large")

        with col_left:
            st.subheader("Location Details")
            st.metric("Latitude", f"{region.lat:.5f}")
            st.metric("Longitude", f"{region.lon:.5f}")
            st.metric("NDVI", f"{region.ndvi:.4f}")
            st.metric("NDWI", f"{region.ndwi:.4f}")
            st.metric("NDBI", f"{region.ndbi:.4f}")
            st.metric("SAVI", f"{region.savi:.4f}")

        with col_right:
            st.metric("Vegetation Health Score", f"{recs['hci']:.2f}")
            st.write("**Overall Assessment:**", recs["habitability"])
            st.write("**Vegetation Status:**", recs["parks"])
            st.write("**Water Presence:**", recs["water"])
            st.write("**Built-up Area:**", recs["builtup"])
            st.caption(
                "Analysis based on real satellite data from Google Earth Engine."
            )

    st.divider()
    st.markdown(
        "### About Vegetation Indices"
        "\n• **NDVI (Normalized Difference Vegetation Index)**: Measures vegetation health and density"
        "\n• **NDWI (Normalized Difference Water Index)**: Detects water bodies and moisture content"
        "\n• **NDBI (Normalized Difference Built-up Index)**: Identifies built-up areas and urban development"
        "\n• **SAVI (Soil-Adjusted Vegetation Index)**: Vegetation index adjusted for soil background"
        "\n\nData source: Copernicus Sentinel-2 satellite imagery via Google Earth Engine"
    )


if __name__ == "__main__":
    main()

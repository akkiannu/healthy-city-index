"""Streamlit version of the Healthy City Index wireframe.

This script mirrors the existing React prototype, providing a Python-first
implementation with mocked indicator data, scoring, and recommendations. The
map section uses pydeck with a blank base layer so the app remains functional
without pulling remote tiles. Future iterations can swap in cached rasters or
custom vector layers once NASA data products are ready.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import rasterio
from rasterio.transform import xy
from rasterio.windows import from_bounds
import streamlit as st


st.set_page_config(page_title="Healthy City Index — Mumbai", layout="wide")

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

# Pre-fill defaults when well-known rasters exist in ./data
DEFAULT_POP_DENSITY_RASTER = DATA_DIR / "ind_pd_2020_1km_UNadj.tif"
DEFAULT_POP_TOTAL_RASTER = DATA_DIR / "ind_pop_2025_CN_1km_R2025A_UA_v1.tif"


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

MUMBAI_BOUNDS = (72.76, 18.89, 73.04, 19.33)  # (west, south, east, north)

DEFAULT_HEATMAP_COLORS = [
    [226, 232, 240],
    [148, 163, 184],
    [71, 85, 105],
]

HEATMAP_COLOR_SCHEMES = {
    "Population density": {
        "colors": [
            [255, 245, 240],
            [254, 224, 210],
            [252, 187, 161],
            [252, 146, 114],
            [251, 106, 74],
            [222, 45, 38],
            [165, 15, 21],
        ],
    },
    "Population total": {
        "colors": [
            [242, 240, 247],
            [218, 218, 235],
            [188, 189, 220],
            [158, 154, 200],
            [128, 125, 186],
            [106, 81, 163],
            [74, 20, 134],
        ],
    },
    "Tree cover": {
        "colors": [
            [237, 248, 233],
            [199, 233, 192],
            [161, 217, 155],
            [116, 196, 118],
            [65, 171, 93],
            [35, 139, 69],
            [0, 90, 50],
        ],
    },
}


def _generate_click_grid(
    bounds: Tuple[float, float, float, float], step: float = 0.01
) -> pd.DataFrame:
    west, south, east, north = bounds
    lat_values = np.arange(south, north + step, step)
    lon_values = np.arange(west, east + step, step)
    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    df = pd.DataFrame(
        {
            "lon": lon_grid.ravel(),
            "lat": lat_grid.ravel(),
        }
    )
    df["lat_display"] = df["lat"].map(lambda v: f"{v:.4f}")
    df["lon_display"] = df["lon"].map(lambda v: f"{v:.4f}")
    return df


MAP_CLICK_GRID = _generate_click_grid(MUMBAI_BOUNDS)

TREECOVER_DIR = DATA_DIR / "treecover"
DEFAULT_TREECOVER_RASTER = None
if TREECOVER_DIR.exists():
    for candidate in ["10N_070E.tif", "20N_070E.tif", "30N_070E.tif"]:
        path = TREECOVER_DIR / candidate
        if path.exists():
            DEFAULT_TREECOVER_RASTER = path
            break


def _normalize_path(path: str) -> Optional[str]:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if not candidate.exists() and not candidate.is_absolute():
        alt = DATA_DIR / candidate
        if alt.exists():
            candidate = alt
    try:
        resolved = candidate.resolve(strict=False)
    except FileNotFoundError:
        return None
    return str(resolved)


@st.cache_resource(show_spinner=False)
def _load_raster(path: str):
    return rasterio.open(path)


@st.cache_data(show_spinner=False)
def _sample_raster(path: str, lat: float, lon: float) -> Optional[float]:
    dataset = _load_raster(path)
    try:
        sample = next(dataset.sample([(lon, lat)]))[0]
    except StopIteration:
        return None
    except Exception:
        raise
    if dataset.nodata is not None and sample == dataset.nodata:
        return None
    if np.isnan(sample):
        return None
    return float(sample)


def _raster_value_from_path(
    path: str, lat: float, lon: float
) -> Tuple[Optional[float], Optional[str]]:
    normalized = _normalize_path(path)
    if not normalized or not Path(normalized).exists():
        return None, "Raster not found at the provided path."
    try:
        value = _sample_raster(normalized, lat, lon)
    except Exception as exc:
        return None, f"Failed to sample raster: {exc}"
    if value is None:
        return None, "Raster returned no data at this location."
    return value, None


@st.cache_data(show_spinner=False)
def _raster_dataframe(
    path: str,
    bounds: Tuple[float, float, float, float],
    max_points: int = 8000,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    normalized = _normalize_path(path)
    if not normalized or not Path(normalized).exists():
        return None, "Raster not found at the provided path."

    dataset = _load_raster(normalized)
    window = from_bounds(*bounds, dataset.transform)
    data = dataset.read(1, window=window, masked=True)
    if data.size == 0:
        return None, "Raster returned no data inside the map bounds."

    mask = ~data.mask
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None, "Raster returned no valid pixels in the map bounds."

    values = data.data[rows, cols].astype(float)
    if rows.size > max_points:
        idx = np.linspace(0, rows.size - 1, max_points, dtype=int)
        rows = rows[idx]
        cols = cols[idx]
        values = values[idx]

    transform = dataset.window_transform(window)
    xs, ys = xy(transform, rows, cols)
    df = pd.DataFrame({"lon": xs, "lat": ys, "value": values})
    return df, None


def _prepare_heatmap_dataframe(
    df: pd.DataFrame, colors: Optional[list[list[int]]] = None
) -> Tuple[pd.DataFrame, float, float]:
    vmin = float(df["value"].min())
    vmax = float(df["value"].max())
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    normalized = (df["value"] - vmin) / (vmax - vmin)
    dataframe = df.copy()
    palette = np.array(colors if colors else DEFAULT_HEATMAP_COLORS, dtype=float)
    if palette.shape[0] < 2:
        palette = np.vstack([palette, palette])
    stops = np.linspace(0.0, 1.0, palette.shape[0])
    values = normalized.to_numpy()
    color_r = np.interp(values, stops, palette[:, 0])
    color_g = np.interp(values, stops, palette[:, 1])
    color_b = np.interp(values, stops, palette[:, 2])
    dataframe["color_r"] = np.clip(color_r, 0, 255).astype(int)
    dataframe["color_g"] = np.clip(color_g, 0, 255).astype(int)
    dataframe["color_b"] = np.clip(color_b, 0, 255).astype(int)
    dataframe["value_display"] = dataframe["value"].map(lambda v: f"{v:,.2f}")
    dataframe["lat_display"] = dataframe["lat"].map(lambda v: f"{v:.4f}")
    dataframe["lon_display"] = dataframe["lon"].map(lambda v: f"{v:.4f}")
    return dataframe, vmin, vmax


def _noise(lat: float, lon: float, seed: int) -> float:
    return abs(math.sin(lat * 5 + lon * 3 + seed)) % 1.0


def fetch_region_data(
    lat: float,
    lon: float,
    population_density: Optional[float] = None,
) -> RegionData:
    n = [_noise(lat, lon, i) for i in range(1, 8)]
    try:
        region = RegionData(
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
        if population_density is not None:
            region = replace(region, pop_density=float(population_density))
        return region
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


def _map_layers(
    point: Dict[str, float],
    heatmap_df: Optional[pd.DataFrame] = None,
    heatmap_label: str = "",
    heatmap_units: str = "",
    heatmap_color_range: Optional[list[list[int]]] = None,
) -> Tuple[pdk.Deck, Dict[str, float]]:
    view_state = pdk.ViewState(
        latitude=point["lat"], longitude=point["lon"], zoom=11, pitch=0
    )

    # Approximate bounding box for Greater Mumbai (covers island city + suburbs).
    rectangle = [
        [72.76, 18.89],
        [73.04, 18.89],
        [73.04, 19.33],
        [72.76, 19.33],
    ]

    rectangle_layer = pdk.Layer(
        "PolygonLayer",
        data=[{"polygon": rectangle}],
        id="mumbai-boundary",
        get_polygon="polygon",
        get_fill_color=[252, 165, 165, 60],
        get_line_color=[248, 113, 113, 200],
        line_width_min_pixels=1,
    )

    grid_layer = pdk.Layer(
        "ScatterplotLayer",
        data=MAP_CLICK_GRID,
        id="map-click-grid",
        get_position="[lon, lat]",
        get_radius=90,
        get_fill_color="[255, 255, 255, 16]",
        opacity=0.05,
        stroked=False,
        pickable=True,
    )

    layers = [rectangle_layer, grid_layer]

    tooltip_html = "<b>Lat:</b> {lat}<br/><b>Lon:</b> {lon}"
    tooltip_style = {"backgroundColor": "#0f172a", "color": "#f8fafc"}

    if heatmap_df is not None and not heatmap_df.empty:
        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=heatmap_df,
            id="heatmap-layer",
            get_position="[lon, lat]",
            get_weight="value",
            radius_pixels=60,
            aggregation="MEAN",
            color_range=heatmap_color_range,
        )
        hover_layer = pdk.Layer(
            "ScatterplotLayer",
            data=heatmap_df,
            id="heatmap-points",
            get_position="[lon, lat]",
            get_fill_color="[color_r, color_g, color_b, 120]",
            get_line_color="[color_r, color_g, color_b, 200]",
            get_radius=80,
            pickable=True,
            auto_highlight=True,
        )
        layers.extend([heatmap_layer, hover_layer])
        if heatmap_label:
            tooltip_html += f"<br/><b>{heatmap_label}:</b> {{value_display}}"
            if heatmap_units:
                tooltip_html += f" {heatmap_units}"

    point_df = pd.DataFrame([
        {
            "lat": point["lat"],
            "lon": point["lon"],
            "lat_display": f"{point['lat']:.4f}",
            "lon_display": f"{point['lon']:.4f}",
        }
    ])

    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=point_df,
        id="selected-point",
        get_position="[lon, lat]",
        get_radius=220,
        get_fill_color=[17, 24, 39, 255],
        get_line_color=[248, 250, 252, 200],
        pickable=True,
        auto_highlight=True,
        line_width_min_pixels=1,
    )
    layers.append(point_layer)

    deck = pdk.Deck(
        map_style=None,
        initial_view_state=view_state,
        layers=layers,
        tooltip={"html": tooltip_html, "style": tooltip_style},
    )
    return deck, view_state.__dict__


def _first_float(mapping: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _point_from_object(obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    lat = _first_float(obj, ("lat", "Lat", "latitude", "Latitude", "LATITUDE"))
    lon = _first_float(obj, ("lon", "Lon", "longitude", "Longitude", "LONGITUDE"))
    if lat is not None and lon is not None:
        return lat, lon
    position = obj.get("position") or obj.get("coordinates")
    if position and len(position) >= 2:
        try:
            lon_value, lat_value = float(position[0]), float(position[1])
        except (TypeError, ValueError):
            return None
        return lat_value, lon_value
    return None


def _selection_to_point(selection: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    if not selection:
        return None
    event = selection.get("selection")
    if not event:
        return None
    objects_by_layer = event.get("objects")
    if not objects_by_layer:
        return None
    preferred_layers = ["heatmap-points", "map-click-grid", "selected-point", "heatmap-layer"]
    for layer_id in preferred_layers:
        for obj in objects_by_layer.get(layer_id, []):
            point = _point_from_object(obj)
            if point:
                return point
    for obj_list in objects_by_layer.values():
        for obj in obj_list:
            point = _point_from_object(obj)
            if point:
                return point
    return None


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
                    heatmap_df_raw, heatmap_error_message = _raster_dataframe(
                        population_raster_input, MUMBAI_BOUNDS
                    )
                    if heatmap_df_raw is not None:
                        heatmap_colors = HEATMAP_COLOR_SCHEMES["Population density"][
                            "colors"
                        ]
                        heatmap_df_map, vmin, vmax = _prepare_heatmap_dataframe(
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
                    heatmap_df_raw, heatmap_error_message = _raster_dataframe(
                        population_total_input, MUMBAI_BOUNDS
                    )
                    if heatmap_df_raw is not None:
                        heatmap_colors = HEATMAP_COLOR_SCHEMES["Population total"][
                            "colors"
                        ]
                        heatmap_df_map, vmin, vmax = _prepare_heatmap_dataframe(
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
                    heatmap_df_raw, heatmap_error_message = _raster_dataframe(
                        treecover_input, MUMBAI_BOUNDS
                    )
                    if heatmap_df_raw is not None:
                        heatmap_colors = HEATMAP_COLOR_SCHEMES["Tree cover"]["colors"]
                        heatmap_df_map, vmin, vmax = _prepare_heatmap_dataframe(
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

            map_deck, _ = _map_layers(
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
            selected_point = _selection_to_point(selection_state)
            if selected_point:
                selected_lat = _clamp(
                    float(selected_point[0]), MUMBAI_BOUNDS[1], MUMBAI_BOUNDS[3]
                )
                selected_lon = _clamp(
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
        point = st.session_state.point
        population_override = None
        population_total_override = None
        treecover_override = None
        if population_raster_input:
            population_value, population_error = _raster_value_from_path(
                population_raster_input, point["lat"], point["lon"]
            )
            if population_error:
                population_status_placeholder.warning(population_error)
            else:
                population_override = population_value
                population_status_placeholder.success(
                    f"Population density sample: {population_override:,.1f} people/km²"
                )
        if population_total_input:
            pop_total_value, pop_total_error = _raster_value_from_path(
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
            treecover_value, treecover_error = _raster_value_from_path(
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

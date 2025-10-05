"""Visualization helpers and map interaction utilities."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk

from .constants import DEFAULT_HEATMAP_COLORS, MUMBAI_BOUNDS


def _generate_click_grid(
    bounds: Tuple[float, float, float, float], step: float = 0.005
) -> pd.DataFrame:
    west, south, east, north = bounds
    lat_values = np.arange(south, north + step, step)
    lon_values = np.arange(west, east + step, step)
    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    df = pd.DataFrame({"lon": lon_grid.ravel(), "lat": lat_grid.ravel()})
    df["lat_display"] = df["lat"].map(lambda v: f"{v:.4f}")
    df["lon_display"] = df["lon"].map(lambda v: f"{v:.4f}")
    return df


MAP_CLICK_GRID = _generate_click_grid(MUMBAI_BOUNDS)


def prepare_heatmap_dataframe(
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


def map_layers(
    point: Dict[str, float],
    heatmap_df: Optional[pd.DataFrame] = None,
    heatmap_label: str = "",
    heatmap_units: str = "",
    heatmap_color_range: Optional[list[list[int]]] = None,
    basemap_tile_url: Optional[str] = None,
    heatmap_opacity: float = 0.75,
) -> Tuple[pdk.Deck, Dict[str, float]]:
    view_state = pdk.ViewState(
        latitude=point["lat"], longitude=point["lon"], zoom=11, pitch=0
    )

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
        get_radius=250,
        radius_min_pixels=12,
        radius_max_pixels=48,
        get_fill_color="[255, 255, 255, 12]",
        opacity=0.06,
        stroked=False,
        pickable=True,
    )

    layers = [rectangle_layer, grid_layer]

    if basemap_tile_url:
        base_layer = pdk.Layer(
            "TileLayer",
            data=basemap_tile_url,
            id="base-map",
            min_zoom=0,
            max_zoom=19,
            tile_size=256,
            opacity=1.0,
            pickable=False,
        )
        layers.insert(0, base_layer)

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
            opacity=max(0.0, min(1.0, heatmap_opacity)),
            pickable=False,
        )
        layers.append(heatmap_layer)

    point_df = pd.DataFrame(
        [
            {
                "lat": point["lat"],
                "lon": point["lon"],
                "lat_display": f"{point['lat']:.4f}",
                "lon_display": f"{point['lon']:.4f}",
            }
        ]
    )

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


def selection_to_point(selection: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
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


__all__ = [
    "MAP_CLICK_GRID",
    "prepare_heatmap_dataframe",
    "map_layers",
    "selection_to_point",
]

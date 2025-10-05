"""NASA POWER API helpers."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Tuple

import pandas as pd
import requests
import streamlit as st

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
REGION_URL = "https://power.larc.nasa.gov/api/temporal/daily/regional"
INVALID_VALUE = -999.0
MIN_REGION_SPAN = 2.0  # degrees


@st.cache_data(show_spinner=False, ttl=12 * 3600)
def fetch_lst(lat: float, lon: float, days: int = 7) -> Tuple[Optional[float], Optional[str]]:
    """Fetch average land surface temperature (earth skin temperature) in °C.

    Parameters
    ----------
    lat, lon: float
        Location of interest in decimal degrees (WGS84).
    days: int
        Number of recent days to average. Defaults to 7.

    Returns
    -------
    (value, message)
        Value in °C if available and an informational message about the source.
    """

    if days <= 0:
        days = 7
    end_date = date.today()
    start_date = end_date - timedelta(days=days - 1)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "parameters": "TS",
        "community": "SB",
        "format": "JSON",
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=20)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - depends on network availability
        return None, f"NASA POWER request failed: {exc}"

    data = response.json()
    parameter = data.get("properties", {}).get("parameter", {}).get("TS", {})
    values = [float(v) for v in parameter.values() if v is not None and v != INVALID_VALUE]

    if not values:
        return None, "NASA POWER returned no skin temperature data for this window."

    average = sum(values) / len(values)
    return round(average, 2), "Skin temperature averaged from NASA POWER (daily TS)."


def _expand_bounds(
    north: float, south: float, east: float, west: float, minimum_span: float = MIN_REGION_SPAN
) -> Tuple[float, float, float, float]:
    """Ensure latitude/longitude spans meet NASA regional API requirements."""

    lat_min, lat_max = sorted((south, north))
    lon_min, lon_max = sorted((west, east))

    if lat_max - lat_min < minimum_span:
        lat_center = (lat_max + lat_min) / 2
        half_span = minimum_span / 2
        lat_min = max(-90.0, lat_center - half_span)
        lat_max = min(90.0, lat_center + half_span)

    if lon_max - lon_min < minimum_span:
        lon_center = (lon_max + lon_min) / 2
        half_span = minimum_span / 2
        lon_min = max(-180.0, lon_center - half_span)
        lon_max = min(180.0, lon_center + half_span)

    return lat_max, lat_min, lon_max, lon_min


@st.cache_data(show_spinner=False, ttl=12 * 3600)
def fetch_lst_heatmap(
    north: float,
    south: float,
    east: float,
    west: float,
    days: int = 7,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fetch a gridded land surface temperature slice suitable for heatmaps."""

    if days <= 0:
        days = 7

    end_date = date.today()
    start_date = end_date - timedelta(days=days - 1)

    req_north, req_south, req_east, req_west = _expand_bounds(north, south, east, west)

    params = {
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "parameters": "TS",
        "community": "SB",
        "latitude-min": req_south,
        "latitude-max": req_north,
        "longitude-min": req_west,
        "longitude-max": req_east,
        "format": "JSON",
    }

    try:
        response = requests.get(REGION_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # pragma: no cover - network dependent
        return None, f"NASA POWER regional request failed: {exc}"

    features = payload.get("features", [])
    if not features:
        return None, "NASA POWER returned no TS features for this window."

    rows = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates", [])
        if len(coords) < 2:
            continue
        lon_val, lat_val = float(coords[0]), float(coords[1])
        if not (south <= lat_val <= north and west <= lon_val <= east):
            continue

        parameter = (
            feature.get("properties", {})
            .get("parameter", {})
            .get("TS", {})
        )
        if not parameter:
            continue

        try:
            values = [float(v) for v in parameter.values() if v is not None and v != INVALID_VALUE]
        except (TypeError, ValueError):
            continue
        if not values:
            continue

        average = sum(values) / len(values)
        rows.append({"lat": lat_val, "lon": lon_val, "value": round(average, 2)})

    if not rows:
        return None, "NASA POWER returned no TS samples within the requested bounds."

    dataframe = pd.DataFrame(rows)
    info = (
        "Land surface temperatures from NASA POWER TS (daily average)"
        f" · {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    return dataframe, info


__all__ = ["fetch_lst", "fetch_lst_heatmap"]

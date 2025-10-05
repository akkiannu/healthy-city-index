"""NASA POWER API helpers."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Tuple

import requests
import streamlit as st

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
INVALID_VALUE = -999.0


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


__all__ = ["fetch_lst"]

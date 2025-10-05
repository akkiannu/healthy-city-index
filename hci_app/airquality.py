"""Air quality helpers backed by the Open-Meteo API."""

from __future__ import annotations

from statistics import mean
from typing import Dict, Optional

import requests
import streamlit as st


API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def _average(values: list[Optional[float]]) -> Optional[float]:
    points = [float(v) for v in values if v is not None]
    if not points:
        return None
    return mean(points)


@st.cache_data(show_spinner=False)
def fetch_air_quality(lat: float, lon: float, hours: int = 24) -> Dict[str, Optional[float]]:
    """Return recent air-quality averages near the given coordinate."""

    try:
        response = requests.get(
            API_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "pm2_5,carbon_monoxide,nitrogen_dioxide,carbon_dioxide",
                "timezone": "UTC",
            },
            timeout=20,
        )
        response.raise_for_status()
    except Exception:
        return {"pm2_5": None, "no2": None, "co": None, "co2": None}

    payload = response.json()
    hourly = payload.get("hourly", {})
    length = len(hourly.get("time", []))
    start_idx = max(0, length - hours)

    def window(key: str) -> list[Optional[float]]:
        series = hourly.get(key, [])
        return series[start_idx:]

    return {
        "pm2_5": _average(window("pm2_5")),
        "no2": _average(window("nitrogen_dioxide")),
        "co": _average(window("carbon_monoxide")),
        "co2": _average(window("carbon_dioxide")),
    }


__all__ = ["fetch_air_quality"]

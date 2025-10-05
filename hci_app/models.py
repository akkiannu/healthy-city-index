"""Data models and mock data generation for the app."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Optional

from .constants import DEFAULT_POINT


@dataclass(frozen=True)
class RegionData:
    lat: float
    lon: float
    air_pm25: float
    air_no2: float  # µmol/m² column
    air_co: float  # µmol/m² column
    air_co2: float  # ppm
    ndvi: float
    ndwi: float
    ndbi: float
    savi: float
    pop_density: float
    lst_c: float
    industrial_km: float


DEFAULT_REGION = RegionData(
    lat=DEFAULT_POINT["lat"],
    lon=DEFAULT_POINT["lon"],
    air_pm25=40.0,
    air_no2=320.0,
    air_co=180.0,
    air_co2=420.0,
    ndvi=0.40,
    ndwi=0.32,
    ndbi=0.28,
    savi=0.35,
    pop_density=12_000,
    lst_c=32.0,
    industrial_km=3.0,
)


def _noise(lat: float, lon: float, seed: int) -> float:
    return abs(math.sin(lat * 5 + lon * 3 + seed)) % 1.0


def _coerce(value: Optional[float], fallback: float) -> float:
    try:
        if value is None:
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def fetch_region_data(
    lat: float,
    lon: float,
    population_density: Optional[float] = None,
    vegetation_indices: Optional[Dict[str, Optional[float]]] = None,
    air_quality: Optional[Dict[str, Optional[float]]] = None,
) -> RegionData:
    """Return mock indicator values for a given coordinate."""

    noise_values = [_noise(lat, lon, i) for i in range(1, 8)]
    mock_nd = round(0.15 + 0.6 * noise_values[3], 2)
    vegetation = {
        "ndvi": mock_nd,
        "ndwi": mock_nd,
        "ndbi": mock_nd,
        "savi": mock_nd,
    }
    if vegetation_indices:
        vegetation = {
            key: _coerce(vegetation_indices.get(key), vegetation[key]) for key in vegetation
        }
    air = {
        "pm2_5": round(15 + 60 * noise_values[0], 1),
        "no2": round(200 + 300 * noise_values[1], 1),
        "co": round(100 + 180 * noise_values[2], 1),
        "co2": round(380 + 60 * noise_values[3], 1),
    }
    if air_quality:
        air = {key: _coerce(air_quality.get(key), air[key]) for key in air}
    try:
        region = RegionData(
            lat=lat,
            lon=lon,
            air_pm25=air["pm2_5"],
            air_no2=air["no2"],
            air_co=air["co"],
            air_co2=air["co2"],
            ndvi=vegetation["ndvi"],
            ndwi=vegetation["ndwi"],
            ndbi=vegetation["ndbi"],
            savi=vegetation["savi"],
            pop_density=int(round(3000 + 25_000 * noise_values[4])),
            lst_c=round(28 + 10 * noise_values[5], 1),
            industrial_km=round(0.1 + 8.0 * noise_values[6], 2),
        )
        if population_density is not None:
            region = replace(region, pop_density=float(population_density))
        return region
    except Exception:
        return DEFAULT_REGION


__all__ = ["RegionData", "DEFAULT_REGION", "fetch_region_data"]

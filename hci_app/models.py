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
    air_no2: float
    water_turbidity: float
    ndvi: float
    pop_density: float
    lst_c: float
    industrial_km: float


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


def fetch_region_data(
    lat: float,
    lon: float,
    population_density: Optional[float] = None,
) -> RegionData:
    """Return mock indicator values for a given coordinate."""

    noise_values = [_noise(lat, lon, i) for i in range(1, 8)]
    try:
        region = RegionData(
            lat=lat,
            lon=lon,
            air_pm25=round(20 + 80 * noise_values[0], 1),
            air_no2=round(10 + 60 * noise_values[1], 1),
            water_turbidity=round(1 + 15 * noise_values[2], 1),
            ndvi=round(0.15 + 0.6 * noise_values[3], 2),
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

"""Scenario simulation utilities for the Streamlit app."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

from .models import RegionData
from .scoring import clamp_value

# Empirical coefficients for how greenery and built-up changes impact metrics.
_COOLING_FACTOR = 7.0
_HEATING_FACTOR = 5.0
_PM25_FACTOR_GREEN = 30.0
_PM25_FACTOR_BUILT = 22.0
_NO2_FACTOR_GREEN = 170.0
_NO2_FACTOR_BUILT = 130.0
_CO_FACTOR_GREEN = 95.0
_CO_FACTOR_BUILT = 85.0
_CO2_FACTOR_GREEN = 18.0
_CO2_FACTOR_BUILT = 12.0
_WATER_FACTOR_GREEN = 0.26
_WATER_FACTOR_BUILT = 0.18
_NDWI_FACTOR_GREEN = 0.45
_NDWI_FACTOR_BUILT = 0.25
_SAVI_FACTOR_GREEN = 0.55
_SAVI_FACTOR_BUILT = 0.30


def simulate_urban_scenario(
    base_region: RegionData, ndvi_target: float, ndbi_target: float
) -> RegionData:
    """Return a RegionData instance adjusted to represent a what-if scenario."""

    ndvi_clamped = clamp_value(ndvi_target, 0.0, 0.9)
    ndbi_clamped = clamp_value(ndbi_target, -0.2, 0.7)

    delta_green = ndvi_clamped - base_region.ndvi
    delta_built = ndbi_clamped - base_region.ndbi

    lst_sim = clamp_value(
        base_region.lst_c - _COOLING_FACTOR * delta_green + _HEATING_FACTOR * delta_built,
        18.0,
        45.0,
    )
    pm25_sim = clamp_value(
        base_region.air_pm25 - _PM25_FACTOR_GREEN * delta_green + _PM25_FACTOR_BUILT * delta_built,
        5.0,
        200.0,
    )
    no2_sim = clamp_value(
        base_region.air_no2 - _NO2_FACTOR_GREEN * delta_green + _NO2_FACTOR_BUILT * delta_built,
        5.0,
        500.0,
    )
    co_sim = clamp_value(
        base_region.air_co - _CO_FACTOR_GREEN * delta_green + _CO_FACTOR_BUILT * delta_built,
        5.0,
        500.0,
    )
    co2_sim = clamp_value(
        base_region.air_co2 - _CO2_FACTOR_GREEN * delta_green + _CO2_FACTOR_BUILT * delta_built,
        350.0,
        520.0,
    )
    water_sim = clamp_value(
        base_region.water_pollution - _WATER_FACTOR_GREEN * delta_green + _WATER_FACTOR_BUILT * delta_built,
        0.6,
        1.8,
    )
    ndwi_sim = clamp_value(
        base_region.ndwi + _NDWI_FACTOR_GREEN * delta_green - _NDWI_FACTOR_BUILT * delta_built,
        -0.4,
        1.0,
    )
    savi_sim = clamp_value(
        base_region.savi + _SAVI_FACTOR_GREEN * delta_green - _SAVI_FACTOR_BUILT * delta_built,
        -1.0,
        1.0,
    )

    return replace(
        base_region,
        ndvi=round(ndvi_clamped, 3),
        ndbi=round(ndbi_clamped, 3),
        ndwi=round(ndwi_sim, 3),
        savi=round(savi_sim, 3),
        lst_c=round(lst_sim, 1),
        air_pm25=round(pm25_sim, 1),
        air_no2=round(no2_sim, 1),
        air_co=round(co_sim, 1),
        air_co2=round(co2_sim, 1),
        water_pollution=round(water_sim, 3),
    )


__all__ = ["simulate_urban_scenario"]

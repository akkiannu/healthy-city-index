"""Scoring and recommendation logic for the Healthy City Index."""

from __future__ import annotations

from typing import Dict

from .models import RegionData


def clamp_value(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _minmax(value: float, minimum: float, maximum: float, invert: bool = False) -> float:
    clamped = clamp_value(value, minimum, maximum)
    score = (clamped - minimum) / (maximum - minimum or 1)
    return 1 - score if invert else score


def compute_scores(data: RegionData) -> Dict[str, float]:
    return {
        "air": 0.5 * _minmax(data.air_pm25, 10, 100, invert=True)
        + 0.5 * _minmax(data.air_no2, 5, 80, invert=True),
        "water": _minmax(data.water_pollution, 0.85, 1.30, invert=True),
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
    value = sum(scores[key] * weights[key] for key in weights)
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
            data.water_pollution > 1.15,
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


__all__ = [
    "clamp_value",
    "compute_scores",
    "composite",
    "recommendations",
]

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
    air_score = 0.5 * _minmax(data.air_pm25, 10, 100, invert=True) + 0.5 * _minmax(
        data.air_no2, 5, 80, invert=True
    )
    water_score = _minmax(data.water_pollution, 0.85, 1.30, invert=True)
    green_score = _minmax(data.ndvi, 0.1, 0.8)
    built_score = _minmax(data.ndbi, -0.1, 0.6, invert=True)
    return {
        "air": air_score,
        "water": water_score,
        "green": green_score,
        "built": built_score,
    }


def composite(scores: Dict[str, float]) -> float:
    weights = {
        "air": 0.28,
        "water": 0.22,
        "green": 0.25,
        "built": 0.25,
    }
    value = sum(scores[key] * weights.get(key, 0) for key in scores)
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

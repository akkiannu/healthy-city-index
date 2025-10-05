"""Shared constants for the Healthy City Index Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

DEFAULT_POP_DENSITY_RASTER = DATA_DIR / "ind_pd_2020_1km_UNadj.tif"
DEFAULT_POP_TOTAL_RASTER = DATA_DIR / "ind_pop_2025_CN_1km_R2025A_UA_v1.tif"

TREECOVER_DIR = DATA_DIR / "treecover"
TREECOVER_CANDIDATES = ["20N_070E.tif"]


def _default_treecover_raster() -> Optional[Path]:
    if not TREECOVER_DIR.exists():
        return None
    for candidate in TREECOVER_CANDIDATES:
        path = TREECOVER_DIR / candidate
        if path.exists():
            return path
    return None


DEFAULT_TREECOVER_RASTER = _default_treecover_raster()

MUMBAI_BOUNDS = (72.76, 18.89, 73.04, 19.33)  # (west, south, east, north)

DEFAULT_POINT = {"lat": 19.0760, "lon": 72.8777}

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
        "legend": "Low = fewer residents · High = denser neighbourhoods",
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
        "legend": "Low = smaller totals · High = larger catchments",
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
        "legend": "Low = sparse canopy · High = richer cover",
    },
    "Water pollution": {
        "colors": [
            [30, 64, 175],
            [37, 99, 235],
            [59, 130, 246],
            [147, 197, 253],
            [250, 204, 21],
            [248, 113, 113],
            [220, 38, 38],
        ],
        "legend": "Low = clearer water · High = turbid/polluted",
    },
    "Built index": {
        "colors": [
            [74, 20, 134],
            [122, 36, 180],
            [190, 24, 105],
            [239, 68, 68],
            [249, 115, 22],
            [253, 186, 116],
            [255, 255, 178],
        ],
        "legend": "Low = softer landscapes · High = hardscape heavy",
    },
    "Land surface temp": {
        "colors": [
            [15, 118, 110],
            [45, 197, 168],
            [125, 211, 252],
            [253, 224, 71],
            [249, 115, 22],
            [220, 38, 38],
            [136, 19, 55],
        ],
        "legend": "Low = cooler skin temps · High = hotter surfaces",
    },
}


PILLAR_DISPLAY = {
    "air": "Air",
    "water": "Water",
    "green": "Green",
    "built": "Built index",
}


__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "DEFAULT_POP_DENSITY_RASTER",
    "DEFAULT_POP_TOTAL_RASTER",
    "DEFAULT_TREECOVER_RASTER",
    "MUMBAI_BOUNDS",
    "DEFAULT_POINT",
    "DEFAULT_HEATMAP_COLORS",
    "HEATMAP_COLOR_SCHEMES",
    "PILLAR_DISPLAY",
]

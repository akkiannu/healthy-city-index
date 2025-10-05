"""Raster helpers with Streamlit caching wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.windows import Window, from_bounds
import streamlit as st

from .constants import DATA_DIR


def normalize_path(path: str) -> Optional[str]:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if not candidate.exists() and not candidate.is_absolute():
        alt = DATA_DIR / candidate
        if alt.exists():
            candidate = alt
    try:
        resolved = candidate.resolve(strict=False)
    except FileNotFoundError:
        return None
    return str(resolved)


@st.cache_resource(show_spinner=False)
def load_raster(path: str):
    return rasterio.open(path)


@st.cache_data(show_spinner=False)
def sample_raster(path: str, lat: float, lon: float) -> Optional[float]:
    dataset = load_raster(path)
    try:
        sample = next(dataset.sample([(lon, lat)]))[0]
    except StopIteration:
        sample = None
    except Exception:
        raise
    if sample is not None:
        if dataset.nodata is not None and sample == dataset.nodata:
            sample = None
        elif np.isnan(sample):
            sample = None
    if sample is not None:
        return float(sample)

    try:
        row, col = dataset.index(lon, lat)
    except Exception:
        return None
    row_i = int(round(row))
    col_i = int(round(col))
    if row_i < 0 or row_i >= dataset.height or col_i < 0 or col_i >= dataset.width:
        return None
    return _nearest_valid_sample(dataset, row_i, col_i)


def _nearest_valid_sample(dataset, row: int, col: int, max_radius: int = 10) -> Optional[float]:
    height, width = dataset.height, dataset.width
    for radius in range(1, max_radius + 1):
        row_start = max(0, row - radius)
        row_stop = min(height, row + radius + 1)
        col_start = max(0, col - radius)
        col_stop = min(width, col + radius + 1)
        if row_start >= row_stop or col_start >= col_stop:
            continue
        window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
        data = dataset.read(1, window=window, masked=True)
        mask = getattr(data, "mask", None)
        valid_mask = (~mask) if mask is not None else np.ones_like(data, dtype=bool)
        if not valid_mask.any():
            continue
        valid_rows, valid_cols = np.where(valid_mask)
        if valid_rows.size == 0:
            continue
        global_rows = valid_rows + row_start
        global_cols = valid_cols + col_start
        distances = np.hypot(global_rows - row, global_cols - col)
        idx = int(np.argmin(distances))
        value = data.data[valid_rows[idx], valid_cols[idx]]
        if dataset.nodata is not None and value == dataset.nodata:
            continue
        if np.isnan(value):
            continue
        return float(value)
    return None


def raster_value_from_path(
    path: str, lat: float, lon: float
) -> Tuple[Optional[float], Optional[str]]:
    normalized = normalize_path(path)
    if not normalized or not Path(normalized).exists():
        return None, "Raster not found at the provided path."
    try:
        value = sample_raster(normalized, lat, lon)
    except Exception as exc:
        return None, f"Failed to sample raster: {exc}"
    if value is None:
        return None, "Raster returned no data at this location."
    return value, None


@st.cache_data(show_spinner=False)
def raster_dataframe(
    path: str,
    bounds: Tuple[float, float, float, float],
    max_points: int = 8000,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    normalized = normalize_path(path)
    if not normalized or not Path(normalized).exists():
        return None, "Raster not found at the provided path."

    dataset = load_raster(normalized)
    window = from_bounds(*bounds, dataset.transform)
    data = dataset.read(1, window=window, masked=True)
    if data.size == 0:
        return None, "Raster returned no data inside the map bounds."

    mask = ~data.mask
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None, "Raster returned no valid pixels in the map bounds."

    values = data.data[rows, cols].astype(float)
    if rows.size > max_points:
        idx = np.linspace(0, rows.size - 1, max_points, dtype=int)
        rows = rows[idx]
        cols = cols[idx]
        values = values[idx]

    transform = dataset.window_transform(window)
    xs, ys = xy(transform, rows, cols)
    df = pd.DataFrame({"lon": xs, "lat": ys, "value": values})
    return df, None


__all__ = [
    "normalize_path",
    "sample_raster",
    "load_raster",
    "raster_value_from_path",
    "raster_dataframe",
]

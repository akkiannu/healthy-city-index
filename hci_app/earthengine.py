"""Google Earth Engine integration helpers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st

from .constants import ROOT_DIR

try:  # pragma: no cover - earthengine is optional in dev environments
    import ee  # type: ignore
except ImportError:  # pragma: no cover
    ee = None  # type: ignore


DEFAULT_CREDENTIALS_PATH = ROOT_DIR / "credentials.json"
DEFAULT_DATE_RANGE = ("2024-10-01", "2025-01-01")
DEFAULT_CLOUD_COVER = 20
class EarthEngineUnavailable(RuntimeError):
    """Raised when Google Earth Engine cannot be used."""


def _load_service_account(path: Path) -> Tuple[str, Path]:
    data = json.loads(path.read_text())
    service_account = data.get("client_email")
    if not service_account:
        raise EarthEngineUnavailable("Service account email missing in credentials.json")
    return service_account, path


@st.cache_resource(show_spinner=False)
def initialise(credentials_path: Path = DEFAULT_CREDENTIALS_PATH) -> Tuple[bool, Optional[str]]:
    """Initialise the Earth Engine client once per session."""

    if ee is None:
        return False, "earthengine-api is not installed."

    resolved = credentials_path.resolve()
    if not resolved.exists():
        return False, f"Credentials file not found at {resolved}"

    try:
        service_account, key_path = _load_service_account(resolved)
        credentials = ee.ServiceAccountCredentials(service_account, str(key_path))  # type: ignore[attr-defined]
        ee.Initialize(credentials=credentials)
        return True, None
    except Exception as exc:  # pragma: no cover - relies on external service
        return False, str(exc)


def is_available() -> bool:
    status, _ = initialise()
    return status


def _geometry_from_bounds(north: float, south: float, east: float, west: float):
    return ee.Geometry.Polygon(
        [
            [east, north],
            [east, south],
            [west, south],
            [west, north],
            [east, north],
        ]
    )


@lru_cache(maxsize=128)
def _cached_indices(
    north: float,
    south: float,
    east: float,
    west: float,
    start_date: str,
    end_date: str,
    max_cloud: int,
    credentials_path: Path,
) -> Dict[str, Optional[float]]:
    status, error = initialise(credentials_path)
    if not status:
        raise EarthEngineUnavailable(error or "Earth Engine unavailable")

    aoi = _geometry_from_bounds(north, south, east, west)
    image = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .median()
    )

    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    savi = image.expression(
        "1.5 * (NIR - RED) / (NIR + RED + 0.5)", {"NIR": image.select("B8"), "RED": image.select("B4")}
    ).rename("SAVI")

    final_image = ndvi.addBands(ndwi).addBands(ndbi).addBands(savi)
    stats = final_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=30)
    results = stats.getInfo()

    return {
        "ndvi": results.get("NDVI"),
        "ndwi": results.get("NDWI"),
        "ndbi": results.get("NDBI"),
        "savi": results.get("SAVI"),
    }


@st.cache_data(show_spinner=False)
def vegetation_indices(
    north: float,
    south: float,
    east: float,
    west: float,
    start_date: str = DEFAULT_DATE_RANGE[0],
    end_date: str = DEFAULT_DATE_RANGE[1],
    max_cloud: int = DEFAULT_CLOUD_COVER,
    credentials_path: Optional[Path] = None,
) -> Dict[str, Optional[float]]:
    path = credentials_path or DEFAULT_CREDENTIALS_PATH
    return _cached_indices(north, south, east, west, start_date, end_date, max_cloud, path.resolve())


def _mean_from_dataset(
    collection_id: str,
    band: str,
    aoi,
    start_date: str,
    end_date: str,
    scale: int,
    statistic: str = "mean",
    extra_filter: Optional[ee.Filter] = None,
):
    collection = ee.ImageCollection(collection_id).filterBounds(aoi).filterDate(start_date, end_date)
    if extra_filter is not None:
        collection = collection.filter(extra_filter)
    try:
        image = collection.mean() if statistic == "mean" else collection.median()
        reducer = ee.Reducer.mean() if statistic == "mean" else ee.Reducer.median()
        stats = image.select(band).reduceRegion(
            reducer=reducer,
            geometry=aoi,
            scale=scale,
            bestEffort=True,
            maxPixels=1_000_000_000,
            tileScale=4,
        )
        info = stats.getInfo()
        if not info:
            return None
        return info.get(band)
    except Exception:
        return None


@lru_cache(maxsize=128)
def _cached_air_quality(
    north: float,
    south: float,
    east: float,
    west: float,
    start_date: str,
    end_date: str,
    max_cloud: int,
    credentials_path: Path,
) -> Dict[str, Optional[float]]:
    status, error = initialise(credentials_path)
    if not status:
        raise EarthEngineUnavailable(error or "Earth Engine unavailable")

    aoi = _geometry_from_bounds(north, south, east, west)
    results: Dict[str, Optional[float]] = {"no2": None, "pm2_5": None, "co": None, "co2": None}

    datasets = [
        (
            "no2",
            "COPERNICUS/S5P/OFFL/L3_NO2",
            "tropospheric_NO2_column_number_density",
            10_000,
            lambda v: v * 1e6 if v is not None else None,  # to µmol/m²
            ee.Filter.lt("CLOUD_FRACTION", 0.3),
        ),
        (
            "co",
            "COPERNICUS/S5P/OFFL/L3_CO",
            "CO_column_number_density",
            10_000,
            lambda v: v * 1e6 if v is not None else None,  # to µmol/m²
            ee.Filter.lt("CLOUD_FRACTION", 0.3),
        ),
        (
            "pm2_5",
            "NASA/GEOS-CF/GEOS-CF_v1FP",
            "pm2_5",
            25_000,
            lambda v: v,
            None,
        ),
        (
            "co2",
            "ODP/OCO2/XCO2_Monthly",
            "xco2",
            50_000,
            lambda v: v,
            None,
        ),
    ]

    for key, collection_id, band, scale, transform, extra_filter in datasets:
        value = _mean_from_dataset(
            collection_id,
            band,
            aoi,
            start_date,
            end_date,
            scale,
            statistic="mean",
            extra_filter=extra_filter,
        )
        if value is None and key == "pm2_5":
            value = _mean_from_dataset(
                "projects/sat-io/open-datasets/air-quality/merra2/pm2_5",
                "PM2P5",
                aoi,
                start_date,
                end_date,
                25_000,
            )
        results[key] = transform(value) if transform else value

    return results


@st.cache_data(show_spinner=False)
def air_quality_indices(
    north: float,
    south: float,
    east: float,
    west: float,
    start_date: str = DEFAULT_DATE_RANGE[0],
    end_date: str = DEFAULT_DATE_RANGE[1],
    max_cloud: int = DEFAULT_CLOUD_COVER,
    credentials_path: Optional[Path] = None,
) -> Dict[str, Optional[float]]:
    path = credentials_path or DEFAULT_CREDENTIALS_PATH
    return _cached_air_quality(north, south, east, west, start_date, end_date, max_cloud, path.resolve())


__all__ = [
    "EarthEngineUnavailable",
    "DEFAULT_CREDENTIALS_PATH",
    "DEFAULT_DATE_RANGE",
    "DEFAULT_CLOUD_COVER",
    "initialise",
    "is_available",
    "vegetation_indices",
    "air_quality_indices",
]

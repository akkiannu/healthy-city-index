"""Google Earth Engine integration helpers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from .constants import ROOT_DIR

try:  # pragma: no cover - earthengine is optional in dev environments
    import ee  # type: ignore
except ImportError:  # pragma: no cover
    ee = None  # type: ignore


DEFAULT_CREDENTIALS_PATH = ROOT_DIR / "credentials.json"
DEFAULT_DATE_RANGE = ("2024-10-01", "2025-01-01")
DEFAULT_CLOUD_COVER = 20
DEFAULT_WATER_DATE_RANGE = ("2024-12-01", "2025-03-01")


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
    extra_filter: Optional[Any] = None,
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


@lru_cache(maxsize=128)
def _cached_water_pollution(
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

    sentinel = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .median()
    )

    swir_ratio_value: Optional[float] = None
    try:
        swir_ratio = sentinel.select("B11").divide(sentinel.select("B12"))
        swir_ratio_value = swir_ratio.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=60, bestEffort=True, maxPixels=1_000_000_000
        ).get("B11")
        swir_ratio_value = float(swir_ratio_value.getInfo()) if swir_ratio_value is not None else None
    except Exception:
        swir_ratio_value = None

    landsat = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"))
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
        .median()
    )

    tir_anomaly_value: Optional[float] = None
    try:
        tir = landsat.select("ST_B10").multiply(0.00341802).add(149.0)
        mean_temp_value = tir.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=90, bestEffort=True, maxPixels=1_000_000_000
        ).get("ST_B10")
        mean_temp_value = (
            float(mean_temp_value.getInfo()) if mean_temp_value is not None else None
        )
        if mean_temp_value is not None:
            anomaly = tir.subtract(mean_temp_value)
            tir_anomaly_value = anomaly.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=aoi, scale=90, bestEffort=True, maxPixels=1_000_000_000
            ).get("ST_B10")
            tir_anomaly_value = (
                float(tir_anomaly_value.getInfo()) if tir_anomaly_value is not None else None
            )
    except Exception:
        tir_anomaly_value = None

    return {"swir_ratio": swir_ratio_value, "tir_anomaly": tir_anomaly_value}


@st.cache_data(show_spinner=False)
def water_pollution_indices(
    north: float,
    south: float,
    east: float,
    west: float,
    start_date: str = DEFAULT_WATER_DATE_RANGE[0],
    end_date: str = DEFAULT_WATER_DATE_RANGE[1],
    max_cloud: int = DEFAULT_CLOUD_COVER,
    credentials_path: Optional[Path] = None,
) -> Dict[str, Optional[float]]:
    path = credentials_path or DEFAULT_CREDENTIALS_PATH
    return _cached_water_pollution(north, south, east, west, start_date, end_date, max_cloud, path.resolve())


@lru_cache(maxsize=32)
def _cached_water_pollution_map(
    north: float,
    south: float,
    east: float,
    west: float,
    start_date: str,
    end_date: str,
    max_cloud: int,
    credentials_path: Path,
    scale: int,
    num_pixels: int,
) -> Optional[pd.DataFrame]:
    status, error = initialise(credentials_path)
    if not status:
        raise EarthEngineUnavailable(error or "Earth Engine unavailable")

    aoi = _geometry_from_bounds(north, south, east, west)
    sentinel = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .median()
    )

    try:
        swir_ratio = sentinel.select("B11").divide(sentinel.select("B12")).rename("swir_ratio")
    except Exception:
        return None

    lonlat = ee.Image.pixelLonLat()
    sample_image = swir_ratio.addBands(lonlat)

    try:
        samples = sample_image.sample(
            region=aoi,
            scale=scale,
            projection="EPSG:4326",
            numPixels=num_pixels,
            geometries=True,
            tileScale=4,
        )
        features = samples.getInfo().get("features", [])
    except Exception:
        return None

    if not features:
        return None

    records = []
    for feature in features:
        props = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coords = geometry.get("coordinates")
        value = props.get("swir_ratio")
        if coords is None or value is None:
            continue
        try:
            lon_val, lat_val = float(coords[0]), float(coords[1])
            records.append({"lon": lon_val, "lat": lat_val, "value": float(value)})
        except (TypeError, ValueError):
            continue

    if not records:
        return None

    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def water_pollution_heatmap(
    north: float,
    south: float,
    east: float,
    west: float,
    start_date: str = DEFAULT_WATER_DATE_RANGE[0],
    end_date: str = DEFAULT_WATER_DATE_RANGE[1],
    max_cloud: int = DEFAULT_CLOUD_COVER,
    credentials_path: Optional[Path] = None,
    scale: int = 300,
    num_pixels: int = 4000,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    path = credentials_path or DEFAULT_CREDENTIALS_PATH
    try:
        dataframe = _cached_water_pollution_map(
            north,
            south,
            east,
            west,
            start_date,
            end_date,
            max_cloud,
            path.resolve(),
            scale,
            num_pixels,
        )
    except EarthEngineUnavailable as exc:
        return None, str(exc)
    except Exception as exc:  # pragma: no cover - network dependent
        return None, str(exc)

    if dataframe is None or dataframe.empty:
        return None, "Water pollution dataset returned no samples."
    return dataframe, None


@lru_cache(maxsize=32)
def _cached_built_index_map(
    north: float,
    south: float,
    east: float,
    west: float,
    start_date: str,
    end_date: str,
    max_cloud: int,
    credentials_path: Path,
    scale: int,
    num_pixels: int,
) -> Optional[pd.DataFrame]:
    status, error = initialise(credentials_path)
    if not status:
        raise EarthEngineUnavailable(error or "Earth Engine unavailable")

    aoi = _geometry_from_bounds(north, south, east, west)
    sentinel = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .median()
    )

    try:
        ndbi = sentinel.expression(
            "(SWIR - NIR) / (SWIR + NIR)",
            {"SWIR": sentinel.select("B11"), "NIR": sentinel.select("B8")},
        ).rename("ndbi")
    except Exception:
        return None

    lonlat = ee.Image.pixelLonLat()
    sample_image = ndbi.addBands(lonlat)

    try:
        samples = sample_image.sample(
            region=aoi,
            scale=scale,
            projection="EPSG:4326",
            numPixels=num_pixels,
            geometries=True,
            tileScale=4,
        )
        features = samples.getInfo().get("features", [])
    except Exception:
        return None

    if not features:
        return None

    rows = []
    for feature in features:
        props = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coords = geometry.get("coordinates")
        value = props.get("ndbi")
        if coords is None or value is None:
            continue
        try:
            lon_val, lat_val = float(coords[0]), float(coords[1])
            rows.append({"lon": lon_val, "lat": lat_val, "value": float(value)})
        except (TypeError, ValueError):
            continue

    if not rows:
        return None

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def built_index_heatmap(
    north: float,
    south: float,
    east: float,
    west: float,
    start_date: str = DEFAULT_WATER_DATE_RANGE[0],
    end_date: str = DEFAULT_WATER_DATE_RANGE[1],
    max_cloud: int = DEFAULT_CLOUD_COVER,
    credentials_path: Optional[Path] = None,
    scale: int = 300,
    num_pixels: int = 4000,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    path = credentials_path or DEFAULT_CREDENTIALS_PATH
    try:
        dataframe = _cached_built_index_map(
            north,
            south,
            east,
            west,
            start_date,
            end_date,
            max_cloud,
            path.resolve(),
            scale,
            num_pixels,
        )
    except EarthEngineUnavailable as exc:
        return None, str(exc)
    except Exception as exc:
        return None, str(exc)

    if dataframe is None or dataframe.empty:
        return None, "Built index dataset returned no samples."
    return dataframe, None


@lru_cache(maxsize=32)
def _cached_lst_map(
    north: float,
    south: float,
    east: float,
    west: float,
    start_date: str,
    end_date: str,
    max_cloud: int,
    credentials_path: Path,
    scale: int,
    num_pixels: int,
) -> Optional[pd.DataFrame]:
    status, error = initialise(credentials_path)
    if not status:
        raise EarthEngineUnavailable(error or "Earth Engine unavailable")

    aoi = _geometry_from_bounds(north, south, east, west)
    collection = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .select("LST_Day_1km")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    try:
        lst_kelvin = collection.mean()
    except Exception:
        return None

    # Convert to °C: scale factor 0.02 and subtract 273.15
    lst_celsius = lst_kelvin.multiply(0.02).subtract(273.15).rename("lst_c")
    lonlat = ee.Image.pixelLonLat()
    sample_image = lst_celsius.addBands(lonlat)

    try:
        samples = sample_image.sample(
            region=aoi,
            scale=scale,
            projection="EPSG:4326",
            numPixels=num_pixels,
            geometries=True,
            tileScale=4,
        )
        features = samples.getInfo().get("features", [])
    except Exception:
        return None

    rows = []
    for feature in features:
        props = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coords = geometry.get("coordinates")
        value = props.get("lst_c")
        if coords is None or value is None:
            continue
        try:
            lon_val, lat_val = float(coords[0]), float(coords[1])
            rows.append({"lon": lon_val, "lat": lat_val, "value": float(value)})
        except (TypeError, ValueError):
            continue

    if not rows:
        return None

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def lst_heatmap(
    north: float,
    south: float,
    east: float,
    west: float,
    start_date: str = DEFAULT_WATER_DATE_RANGE[0],
    end_date: str = DEFAULT_WATER_DATE_RANGE[1],
    max_cloud: int = DEFAULT_CLOUD_COVER,
    credentials_path: Optional[Path] = None,
    scale: int = 1000,
    num_pixels: int = 3000,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    path = credentials_path or DEFAULT_CREDENTIALS_PATH
    try:
        dataframe = _cached_lst_map(
            north,
            south,
            east,
            west,
            start_date,
            end_date,
            max_cloud,
            path.resolve(),
            scale,
            num_pixels,
        )
    except EarthEngineUnavailable as exc:
        return None, str(exc)
    except Exception as exc:
        return None, str(exc)

    if dataframe is None or dataframe.empty:
        return None, "LST dataset returned no samples."
    return dataframe, None


@lru_cache(maxsize=256)
def _cached_lst_point(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    credentials_path: Path,
    scale: int,
) -> Optional[float]:
    status, error = initialise(credentials_path)
    if not status:
        raise EarthEngineUnavailable(error or "Earth Engine unavailable")

    point = ee.Geometry.Point([lon, lat])
    collection = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .select("LST_Day_1km")
        .filterBounds(point)
        .filterDate(start_date, end_date)
    )

    try:
        lst_kelvin = collection.mean()
    except Exception:
        return None

    lst_celsius = lst_kelvin.multiply(0.02).subtract(273.15).rename("lst_c")

    try:
        feature = (
            lst_celsius.sample(
                region=point.buffer(scale).bounds(),
                scale=scale,
                projection="EPSG:4326",
                numPixels=1,
                tileScale=4,
            )
            .first()
        )
        if feature is None:
            return None
        info = feature.getInfo()
    except Exception:
        return None

    value = (info or {}).get("properties", {}).get("lst_c") if info else None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def lst_at_point(
    lat: float,
    lon: float,
    start_date: str = DEFAULT_WATER_DATE_RANGE[0],
    end_date: str = DEFAULT_WATER_DATE_RANGE[1],
    credentials_path: Optional[Path] = None,
    scale: int = 1000,
) -> Tuple[Optional[float], Optional[str]]:
    path = credentials_path or DEFAULT_CREDENTIALS_PATH
    try:
        value = _cached_lst_point(
            float(lat),
            float(lon),
            start_date,
            end_date,
            path.resolve(),
            scale,
        )
    except EarthEngineUnavailable as exc:
        return None, str(exc)
    except Exception as exc:
        return None, str(exc)

    if value is None:
        return None, "LST dataset returned no sample at this location."

    message = (
        "Skin temperature from MODIS (Earth Engine)"
        f" · {start_date} to {end_date}"
    )
    return value, message


__all__ = [
    "EarthEngineUnavailable",
    "DEFAULT_CREDENTIALS_PATH",
    "DEFAULT_DATE_RANGE",
    "DEFAULT_CLOUD_COVER",
    "initialise",
    "is_available",
    "vegetation_indices",
    "air_quality_indices",
    "water_pollution_indices",
    "water_pollution_heatmap",
    "built_index_heatmap",
    "lst_heatmap",
    "lst_at_point",
]

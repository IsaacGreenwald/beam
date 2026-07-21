"""Weather layer: fetch hourly atmospheric drivers for a location (Open-Meteo)."""

from __future__ import annotations

import pandas as pd
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

MODELS = {
    "ecmwf": "ecmwf_ifs025",
    "gfs": "gfs_seamless",
    "icon": "icon_seamless",
}

# Open-Meteo field -> our column name.
FIELDS = {
    "direct_normal_irradiance": "dni",      # W/m^2
    "diffuse_radiation": "diffuse",         # W/m^2
    "temperature_2m": "temp_air",           # deg C
    "relative_humidity_2m": "rh",           # %
    "wind_speed_10m": "wind",               # m/s
    "cloud_cover": "cloud",                 # %
}


def parse_hourly(hourly: dict) -> pd.DataFrame:
    """Open-Meteo 'hourly' block -> tidy UTC-indexed DataFrame."""
    times = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True), name="time")
    return pd.DataFrame({col: hourly[api] for api, col in FIELDS.items()}, index=times)


def fetch_model(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    model: str = "ecmwf",
) -> pd.DataFrame:
    """Fetch one model's hourly weather for a location. Dates are 'YYYY-MM-DD'."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {list(MODELS)}")
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(FIELDS),
        "models": MODELS[model],
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "start_date": start_date,
        "end_date": end_date,
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    resp.raise_for_status()
    return parse_hourly(resp.json()["hourly"])

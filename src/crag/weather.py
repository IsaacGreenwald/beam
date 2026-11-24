"""Weather layer: fetch hourly atmospheric drivers for a location (Open-Meteo)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from requests_cache import NEVER_EXPIRE, CachedSession

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CACHE_NAME = "crag/open_meteo"  # under the OS user cache dir
CACHE_EXPIRE_SECONDS = 3600
ARCHIVE_STABLE_DAYS = 14  # past this the archive stops revising -> cache forever

_session: CachedSession | None = None


def _get_session() -> CachedSession:
    global _session
    if _session is None:
        _session = CachedSession(
            cache_name=CACHE_NAME,
            backend="sqlite",
            use_cache_dir=True,
            expire_after=CACHE_EXPIRE_SECONDS,
        )
    return _session


MODELS = {
    "ecmwf": "ecmwf_ifs025",
    "gfs": "gfs_seamless",
    "icon": "icon_seamless",
}

# Open-Meteo field -> our column name.
FIELDS = {
    "direct_normal_irradiance": "dni",  # W/m^2
    "diffuse_radiation": "diffuse",  # W/m^2
    "temperature_2m": "temp_air",  # deg C
    "relative_humidity_2m": "rh",  # %
    "wind_speed_10m": "wind",  # m/s
    "cloud_cover": "cloud",  # %
}


_MAGNUS_B = 17.625
_MAGNUS_C = 243.04  # deg C


def dew_point(temp_c, rh_pct):
    """Dew point (deg C) from temperature and relative humidity, via Magnus."""
    gamma = np.log(rh_pct / 100.0) + _MAGNUS_B * temp_c / (_MAGNUS_C + temp_c)
    return _MAGNUS_C * gamma / (_MAGNUS_B - gamma)


def with_dew_point(df: pd.DataFrame) -> pd.DataFrame:
    """Weather table + a 'dew_point' column derived from temp_air and rh."""
    return df.assign(dew_point=dew_point(df["temp_air"], df["rh"]))


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
    resp = _get_session().get(OPEN_METEO_URL, params=params, timeout=30)
    resp.raise_for_status()
    return parse_hourly(resp.json()["hourly"])


def _trim_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leading/trailing rows with missing fields, keeping a contiguous block."""
    ok = df.notna().all(axis=1).to_numpy()
    if not ok.any():
        return df.iloc[:0]
    return df.iloc[ok.argmax() : len(ok) - ok[::-1].argmax()]


def fetch_archive(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    drop_future: bool = True,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """ERA5-backed hourly history for a location, 1940 to `now` (default: the current time).

    drop_future cuts at now's hour: Open-Meteo fills the recent edge with forecasts.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(FIELDS),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "start_date": start_date,
        "end_date": end_date,
    }
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    settled = pd.Timestamp(end_date, tz="UTC") < now - pd.Timedelta(days=ARCHIVE_STABLE_DAYS)
    resp = _get_session().get(
        ARCHIVE_URL,
        params=params,
        timeout=60,
        expire_after=NEVER_EXPIRE if settled else CACHE_EXPIRE_SECONDS,
    )
    resp.raise_for_status()
    df = parse_hourly(resp.json()["hourly"])
    if drop_future:
        df = df[df.index <= now.floor("h")]
    return _trim_edges(df)


def fetch_models(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    models: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch several models; return {model_name: hourly table}. Defaults to all."""
    models = models or list(MODELS)
    return {m: fetch_model(lat, lon, start_date, end_date, model=m) for m in models}

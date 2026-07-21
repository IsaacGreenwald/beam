import pandas as pd
import pytest

from crag import weather
from crag.weather import FIELDS, dew_point, fetch_models, parse_hourly, with_dew_point


def test_parse_hourly_shapes_and_utc():
    hourly = {
        "time": ["2026-07-20T00:00", "2026-07-20T01:00", "2026-07-20T02:00"],
        "direct_normal_irradiance": [0.0, 0.0, 120.0],
        "diffuse_radiation": [0.0, 5.0, 40.0],
        "temperature_2m": [18.0, 17.5, 19.0],
        "relative_humidity_2m": [60, 62, 55],
        "wind_speed_10m": [1.2, 1.0, 2.3],
        "cloud_cover": [10, 0, 25],
    }
    df = parse_hourly(hourly)

    assert list(df.columns) == list(FIELDS.values())
    assert len(df) == 3
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df["dni"].iloc[2] == 120.0


@pytest.mark.parametrize(
    "temp, rh, expected",
    [
        (20.0, 50.0, 9.3),   # standard reference
        (25.0, 100.0, 25.0),  # saturated air: dew point == temperature
        (30.0, 20.0, 4.6),   # hot and dry: dew point well below air
    ],
)
def test_dew_point_known_values(temp, rh, expected):
    assert dew_point(temp, rh) == pytest.approx(expected, abs=0.2)


def test_dew_point_never_exceeds_temperature():
    for temp in range(-10, 45, 5):
        for rh in (10, 40, 70, 99, 100):
            assert dew_point(temp, rh) <= temp + 1e-6


def test_with_dew_point_adds_column():
    hourly = {
        "time": ["2026-07-20T00:00", "2026-07-20T01:00"],
        "direct_normal_irradiance": [0.0, 0.0],
        "diffuse_radiation": [0.0, 0.0],
        "temperature_2m": [20.0, 25.0],
        "relative_humidity_2m": [50.0, 100.0],
        "wind_speed_10m": [1.0, 1.0],
        "cloud_cover": [0, 0],
    }
    df = with_dew_point(parse_hourly(hourly))
    assert "dew_point" in df.columns
    assert df["dew_point"].iloc[0] == pytest.approx(9.3, abs=0.2)
    assert df["dew_point"].iloc[1] == pytest.approx(25.0, abs=0.2)


def test_fetch_models_aggregates_per_model(monkeypatch):
    calls = []

    def fake_fetch_model(lat, lon, start_date, end_date, model="ecmwf"):
        calls.append(model)
        return pd.DataFrame({"dni": [float(len(model))]})

    monkeypatch.setattr(weather, "fetch_model", fake_fetch_model)
    out = fetch_models(40.0, -105.0, "2026-07-20", "2026-07-20", models=["ecmwf", "gfs"])

    assert set(out) == {"ecmwf", "gfs"}
    assert calls == ["ecmwf", "gfs"]
    assert isinstance(out["ecmwf"], pd.DataFrame)


def test_session_is_cached_singleton():
    import requests_cache

    s1 = weather._get_session()
    s2 = weather._get_session()
    assert s1 is s2
    assert isinstance(s1, requests_cache.CachedSession)

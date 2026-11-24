import numpy as np
import pandas as pd
import pytest
from requests_cache.backends import sqlite as requests_cache_sqlite

from crag import weather
from crag.weather import (
    FIELDS,
    dew_point,
    fetch_archive,
    fetch_models,
    parse_hourly,
    with_dew_point,
)


class _FakeResponse:
    def __init__(self, hourly):
        self._hourly = hourly

    def raise_for_status(self):
        pass

    def json(self):
        return {"hourly": self._hourly}


class _FakeSession:
    """Captures the outgoing request and replays a canned hourly block."""

    def __init__(self, hourly):
        self._hourly = hourly
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return _FakeResponse(self._hourly)


def _hourly_block(times, **overrides):
    n = len(times)
    block = {"time": list(times)}
    for api in FIELDS:
        block[api] = list(overrides.get(api, [1.0] * n))
    block["relative_humidity_2m"] = list(overrides.get("relative_humidity_2m", [50.0] * n))
    return block


def _install(monkeypatch, hourly):
    session = _FakeSession(hourly)
    monkeypatch.setattr(weather, "_get_session", lambda: session)
    return session


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
        (20.0, 50.0, 9.3),  # standard reference
        (25.0, 100.0, 25.0),  # saturated air: dew point == temperature
        (30.0, 20.0, 4.6),  # hot and dry: dew point well below air
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


def test_fetch_archive_request_shape(monkeypatch):
    times = pd.date_range("2016-07-01", periods=6, freq="h").strftime("%Y-%m-%dT%H:%M")
    session = _install(monkeypatch, _hourly_block(times))
    df = fetch_archive(40.0, -105.0, "2016-07-01", "2016-07-01")

    call = session.calls[0]
    assert call["url"] == weather.ARCHIVE_URL
    # km/h is the archive default; without this convection is overestimated 3.6x
    assert call["params"]["wind_speed_unit"] == "ms"
    assert call["params"]["timezone"] == "GMT"
    assert call["params"]["hourly"] == ",".join(FIELDS)
    assert list(df.columns) == list(FIELDS.values())
    assert str(df.index.tz) == "UTC"


def test_fetch_archive_caches_settled_ranges_forever(monkeypatch):
    times = pd.date_range("2016-07-01", periods=3, freq="h").strftime("%Y-%m-%dT%H:%M")
    session = _install(monkeypatch, _hourly_block(times))

    fetch_archive(40.0, -105.0, "2016-07-01", "2016-07-01")
    assert session.calls[0]["expire_after"] == weather.NEVER_EXPIRE

    recent = pd.Timestamp.now(tz="UTC").date().isoformat()
    fetch_archive(40.0, -105.0, recent, recent)
    assert session.calls[1]["expire_after"] == weather.CACHE_EXPIRE_SECONDS


def test_fetch_archive_drops_forecast_backed_future_hours(monkeypatch):
    now = pd.Timestamp.now(tz="UTC").floor("h")
    times = pd.date_range(now - pd.Timedelta(hours=3), periods=8, freq="h")
    session = _install(monkeypatch, _hourly_block(times.strftime("%Y-%m-%dT%H:%M")))

    df = fetch_archive(40.0, -105.0, "2026-01-01", "2026-01-02")
    assert df.index.max() <= now
    assert len(df) == 4  # now-3h .. now inclusive

    kept = fetch_archive(40.0, -105.0, "2026-01-01", "2026-01-02", drop_future=False)
    assert len(kept) == 8
    assert len(session.calls) == 2


def test_fetch_archive_trims_edge_gaps(monkeypatch):
    times = pd.date_range("2016-07-01", periods=6, freq="h").strftime("%Y-%m-%dT%H:%M")
    temps = [None, 15.0, 16.0, 17.0, 18.0, None]  # edges unusable, middle contiguous
    _install(monkeypatch, _hourly_block(times, temperature_2m=temps))

    df = fetch_archive(40.0, -105.0, "2016-07-01", "2016-07-01")
    assert len(df) == 4
    assert not df.isna().any().any()
    assert np.allclose(df["temp_air"].to_numpy(), [15.0, 16.0, 17.0, 18.0])


def test_session_is_a_singleton_in_the_os_cache_dir(monkeypatch, tmp_path):
    os_cache, elsewhere = tmp_path / "os-cache", tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(requests_cache_sqlite, "user_cache_dir", lambda: str(os_cache))
    monkeypatch.setattr(weather, "_session", None)
    monkeypatch.chdir(elsewhere)

    s1 = weather._get_session()
    assert s1 is weather._get_session()
    assert s1.cache.db_path == os_cache / "crag" / "open_meteo.sqlite"

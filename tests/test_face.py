"""Face: time zone derived from coordinates."""

from __future__ import annotations

from crag.face import Face


def _tz(lat: float, lon: float) -> str:
    return Face(id="t", lat=lat, lon=lon, aspect=180).tz


def test_tz_from_coordinates():
    assert _tz(40.0, -105.27) == "America/Denver"
    assert _tz(48.4, 2.7) == "Europe/Paris"
    assert _tz(-33.9, 18.4) == "Africa/Johannesburg"


def test_tz_at_sea_is_still_defined():
    assert _tz(35.0, -140.0) == "Etc/GMT+9"

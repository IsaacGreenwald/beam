"""sun_windows: minute-resolution spans of direct sun on a face."""

from __future__ import annotations

import pandas as pd
import pvlib

from crag.face import Face
from crag.geometry import beam_factor, sun_windows
from crag.horizon import ridge

BOULDER = dict(lat=40.0, lon=-105.27)
EQUINOX = "2026-03-20"
MINUTE = pd.Timedelta(minutes=1)


def _face(aspect: float, **kw) -> Face:
    return Face(id="w", aspect=aspect, **BOULDER, **kw)


def _transit(face: Face, date: str) -> pd.Timestamp:
    noon = pd.DatetimeIndex([pd.Timestamp(f"{date} 12:00", tz=face.tz)])
    spa = pvlib.solarposition.sun_rise_set_transit_spa(noon, face.lat, face.lon)
    return spa["transit"].iloc[0]


def test_south_face_on_equinox_is_lit_all_day_centered_on_solar_noon():
    face = _face(180)
    w = sun_windows(face, EQUINOX)
    assert len(w) == 1
    assert abs(w["duration"].iloc[0] - pd.Timedelta(hours=12)) < pd.Timedelta(minutes=15)
    mid = w["start"].iloc[0] + w["duration"].iloc[0] / 2
    assert abs(mid - _transit(face, EQUINOX)) <= 2 * MINUTE


def test_windows_are_in_the_faces_local_time():
    w = sun_windows(_face(180), EQUINOX)
    assert str(w["start"].iloc[0].tz) == "America/Denver"
    assert w["start"].iloc[0].hour == 7


def test_north_face_on_winter_solstice_gets_no_sun():
    assert sun_windows(_face(0), "2026-12-21").empty


def test_east_face_loses_sun_at_solar_noon():
    face = _face(90)
    w = sun_windows(face, EQUINOX)
    assert len(w) == 1
    assert abs(w["end"].iloc[0] - _transit(face, EQUINOX)) <= 2 * MINUTE


def test_narrow_skyline_bump_splits_the_day():
    face = _face(180, horizon=ridge(180, 60, 5))
    w = sun_windows(face, EQUINOX)
    assert len(w) == 2
    assert w["end"].iloc[0] < _transit(face, EQUINOX) < w["start"].iloc[1]


def test_windows_cover_exactly_the_lit_minutes():
    face = _face(135, horizon=ridge(150, 35, 20))
    w = sun_windows(face, "2026-06-21")
    times = pd.date_range(
        pd.Timestamp("2026-06-21", tz=face.tz), periods=24 * 60, freq="1min"
    )
    lit_minutes = int((beam_factor(face, times) > 0).sum())
    assert w["duration"].sum() == lit_minutes * MINUTE
    assert (w["peak"] > 0).all() and (w["peak"] <= 1).all()


def test_dst_switch_day_reports_daylight_time():
    face = _face(180)
    w = sun_windows(face, "2026-03-08")
    assert len(w) == 1
    assert w["start"].iloc[0].utcoffset() == pd.Timedelta(hours=-6)

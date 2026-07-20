"""Tests for the orientation-only geometry engine (Step 1).

Two flavours:
  * SANITY checks -- physical facts that must hold regardless of any fitting.
    These turn the abstract dot product into "yes, a north face is dark at noon."
  * PROPERTY checks -- invariants that must hold for ANY face/time (hypothesis).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from crag.face import Face
from crag.geometry import beam_factor, sun_position

# A mid-latitude Northern-hemisphere location for the directional sanity checks.
LAT, LON = 40.0, -105.0  # Front Range, Colorado -ish


def _day(date: str, tz: str = "US/Mountain") -> pd.DatetimeIndex:
    """Hourly timestamps across one local day."""
    return pd.date_range(f"{date} 00:00", f"{date} 23:00", freq="h", tz=tz)


def _noon_value(face: Face, date: str) -> float:
    times = _day(date)
    bf = beam_factor(face, times)
    # local solar-ish noon: take the hour of peak sun elevation
    sun = sun_position(face, times)
    noon_idx = sun["elevation"].to_numpy().argmax()
    return float(bf.iloc[noon_idx])


# --- SANITY: south vertical face, winter noon beats summer noon ---
def test_south_face_winter_noon_beats_summer():
    """Low winter sun hits a south vertical face more squarely than high summer sun."""
    south = Face(id="s-vert", lat=LAT, lon=LON, aspect=180, dip=90)
    winter = _noon_value(south, "2026-12-21")
    summer = _noon_value(south, "2026-06-21")
    assert winter > summer, f"winter {winter:.3f} should exceed summer {summer:.3f}"


# --- SANITY: north face is dark at midsummer noon, but catches early/late sun ---
def test_north_face_dark_at_summer_noon_lit_at_edges():
    north = Face(id="n-vert", lat=LAT, lon=LON, aspect=0, dip=90)
    times = _day("2026-06-21")
    bf = beam_factor(north, times).to_numpy()
    sun_el = sun_position(north, times)["elevation"].to_numpy()

    noon_idx = sun_el.argmax()
    assert bf[noon_idx] == 0.0, "north vertical face must be self-shaded at solar noon"
    # but it should be lit at some point (early morning / late evening sun swings north)
    assert bf.max() > 0.0, "north face should catch direct sun at the day's edges in midsummer"


# --- SANITY: night is always zero ---
def test_no_sun_at_night():
    face = Face(id="any", lat=LAT, lon=LON, aspect=180, dip=90)
    # 2 a.m. local, mid-winter -- definitively dark
    times = pd.date_range("2026-12-21 02:00", periods=1, freq="h", tz="US/Mountain")
    assert beam_factor(face, times).iloc[0] == 0.0


# --- PROPERTY: beam_factor always in [0, 1], for any face and any hour ---
@settings(max_examples=50, deadline=None)
@given(
    aspect=st.floats(min_value=0, max_value=360),
    dip=st.floats(min_value=0, max_value=180),
    hour=st.integers(min_value=0, max_value=23),
    month=st.integers(min_value=1, max_value=12),
)
def test_beam_factor_bounded(aspect, dip, hour, month):
    face = Face(id="prop", lat=LAT, lon=LON, aspect=aspect, dip=dip)
    times = pd.date_range(
        f"2026-{month:02d}-15 {hour:02d}:00", periods=1, freq="h", tz="US/Mountain"
    )
    val = beam_factor(face, times).iloc[0]
    assert 0.0 <= val <= 1.0 + 1e-9, f"beam_factor {val} out of [0,1]"
    assert np.isfinite(val)

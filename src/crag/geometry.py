"""When, and how squarely, the sun strikes a face."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from crag.face import Face
from crag.horizon import elevation_at


def _unit_vectors(elevation_deg: np.ndarray, azimuth_deg: np.ndarray) -> np.ndarray:
    """(elevation, azimuth) in degrees, azimuth N=0 clockwise -> (3, N) East-North-Up."""
    el = np.radians(elevation_deg)
    az = np.radians(azimuth_deg)
    return np.vstack(
        [
            np.cos(el) * np.sin(az),  # East
            np.cos(el) * np.cos(az),  # North
            np.sin(el),  # Up
        ]
    )


def sky_view_factor(face: Face) -> float:
    """Fraction of the sky hemisphere the face sees (isotropic): (1+cos(dip))/2."""
    return (1.0 + np.cos(np.radians(face.dip))) / 2.0


def horizon_elevation(face: Face, azimuth_deg: np.ndarray) -> np.ndarray:
    """Interpolated skyline elevation (deg) in each given azimuth direction."""
    return elevation_at(face.horizon, azimuth_deg)


def sun_position(face: Face, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Sun 'elevation' and 'azimuth' (deg, N=0 clockwise) at the face. times must be tz-aware."""
    if times.tz is None:
        raise ValueError("times must be timezone-aware (pvlib silently assumes UTC otherwise)")
    solpos = pvlib.solarposition.get_solarposition(
        times, face.lat, face.lon, altitude=face.altitude
    )
    return pd.DataFrame(
        {
            "elevation": solpos["apparent_elevation"].to_numpy(),
            "azimuth": solpos["azimuth"].to_numpy(),
        },
        index=times,
    )


def beam_factor(face: Face, times: pd.DatetimeIndex) -> pd.Series:
    """cos(angle of incidence) where the sun is up, above the skyline, and in front; else 0.

    Multiply by DNI for W/m^2 on the face.
    """
    sun = sun_position(face, times)
    sun_el = sun["elevation"].to_numpy()
    sun_az = sun["azimuth"].to_numpy()

    normal = _unit_vectors(np.array([face.normal_elevation]), np.array([face.aspect]))  # (3, 1)
    sun_vec = _unit_vectors(sun_el, sun_az)  # (3, N)

    cos_aoi = (normal * sun_vec).sum(axis=0)  # (N,)

    above_skyline = sun_el > horizon_elevation(face, sun_az)
    lit = (cos_aoi > 0) & (sun_el > 0) & above_skyline
    factor = np.where(lit, cos_aoi, 0.0)

    return pd.Series(factor, index=times, name="beam_factor")


def sun_windows(face: Face, date) -> pd.DataFrame:
    """Minute-resolution spans of direct sun on the face over one local day."""
    day = pd.Timestamp(date).normalize()
    start = day.tz_localize(face.tz, nonexistent="shift_forward")
    end = (day + pd.Timedelta(days=1)).tz_localize(face.tz, nonexistent="shift_forward")
    times = pd.date_range(start, end, freq="1min", inclusive="left")
    bf = beam_factor(face, times).to_numpy()

    lit = np.concatenate([[False], bf > 0, [False]])
    starts = np.flatnonzero(~lit[:-1] & lit[1:])
    ends = np.flatnonzero(lit[:-1] & ~lit[1:])
    minutes = times.append(pd.DatetimeIndex([end]))
    return pd.DataFrame(
        {
            "start": minutes[starts],
            "end": minutes[ends],
            "duration": minutes[ends] - minutes[starts],
            "peak": [bf[a:b].max() for a, b in zip(starts, ends, strict=True)],
        }
    )

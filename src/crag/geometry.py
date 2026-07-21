"""The geometry engine: when, and how squarely, does the sun strike a face?

Step 1 scope: ORIENTATION ONLY. We compute the sun's track (via pvlib) and the
face's own self-shading + cosine projection. Terrain/skyline occlusion and diffuse
sky-view come in later steps.

Everything here is a pure function of (Face, times). No I/O, no fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from crag.face import Face


def _unit_vectors(elevation_deg: np.ndarray, azimuth_deg: np.ndarray) -> np.ndarray:
    """Convert (elevation, azimuth) angles to unit vectors in East-North-Up.

    azimuth measured from North, clockwise (N=0, E=90, S=180, W=270).
    Returns shape (3, N): rows are [East, North, Up].
    """
    el = np.radians(elevation_deg)
    az = np.radians(azimuth_deg)
    return np.vstack(
        [
            np.cos(el) * np.sin(az),  # East
            np.cos(el) * np.cos(az),  # North
            np.sin(el),               # Up
        ]
    )


def sky_view_factor(face: Face) -> float:
    """Fraction of the sky hemisphere the face sees (isotropic): (1+cos(dip))/2."""
    return (1.0 + np.cos(np.radians(face.dip))) / 2.0


def horizon_elevation(face: Face, azimuth_deg: np.ndarray) -> np.ndarray:
    """Interpolated skyline elevation (deg) in each given azimuth direction.

    Empty horizon -> 0 everywhere (open sky). Wraps around 360 via np.interp's
    `period`, so a sample at 350 deg and one at 10 deg blend across North correctly.
    """
    az_query = np.asarray(azimuth_deg, dtype=float) % 360.0
    if not face.horizon:
        return np.zeros_like(az_query)
    az = np.array([p[0] for p in face.horizon], dtype=float)
    el = np.array([p[1] for p in face.horizon], dtype=float)
    order = np.argsort(az)
    return np.interp(az_query, az[order], el[order], period=360.0)


def sun_position(face: Face, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Sun elevation and azimuth over time for this face's location.

    Thin wrapper over pvlib. `times` must be timezone-aware. Returns a DataFrame
    with 'elevation' (deg above horizon) and 'azimuth' (deg from North, clockwise).
    """
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
    """Fraction of direct-normal sunlight landing on the face, per timestep.

    Returns a value in [0, 1]:
      * 0  when the sun is below the horizon, OR behind the plane of the face
            (self-shading).
      * cos(angle of incidence) otherwise -- Lambert's cosine projection.

    Multiply this by DNI (direct-normal irradiance) to get W/m^2 on the face.

    Three gates decide whether direct sun reaches the face:
      1. above the true horizon  (sun elevation > 0)
      2. above the skyline       (sun elevation > terrain in that direction)
      3. in front of the face    (cos_aoi > 0, i.e. not self-shaded)
    Where all three hold, the value is cos_aoi (Lambert projection); else 0.
    """
    sun = sun_position(face, times)
    sun_el = sun["elevation"].to_numpy()
    sun_az = sun["azimuth"].to_numpy()

    # The two direction vectors: the outward face normal, and the sun.
    normal = _unit_vectors(
        np.array([face.normal_elevation]), np.array([face.aspect])
    )  # (3, 1)
    sun_vec = _unit_vectors(sun_el, sun_az)  # (3, N)

    # Angle of incidence: dot product of the two unit vectors.
    cos_aoi = (normal * sun_vec).sum(axis=0)  # (N,)

    above_skyline = sun_el > horizon_elevation(face, sun_az)
    lit = (cos_aoi > 0) & (sun_el > 0) & above_skyline
    factor = np.where(lit, cos_aoi, 0.0)

    return pd.Series(factor, index=times, name="beam_factor")

"""Horizon sources: functions that GENERATE a horizon table for a Face.

The engine (`crag.geometry.beam_factor`) only ever *consumes* a horizon table and
does not care how it was produced. This module is the seam where table generators
live, so new sources drop in without touching the physics core. A horizon table is
just a tuple of ``(azimuth_deg, elevation_deg)`` samples -- the shape `Face.horizon`
expects.

Available now:
  * ``flat_horizon`` -- open sky (0 deg everywhere). The default.
  * ``ridge``        -- a simple synthetic block; for tests and quick hand-entry.

Reserved for later (stubs below, so the seam is real and the home is obvious):
  * ``from_elevation_api`` -- raytrace a bare-earth skyline from an elevation API.
  * ``from_lidar_pointcloud`` -- the interesting one. Classify vegetation vs. rock
    (multi-return / ASPRS class codes), build an OPAQUE horizon from the rock
    returns, and derive a canopy-density attenuation layer from the veg returns.
    See BACKLOG.md.

Tables from several sources (DEM skyline + a few hand-entered near-field bumps)
combine with ``merge``, which keeps the tallest skyline in every direction. Never
concatenate tables directly: one table's taper-to-zero samples would cut into the
other's terrain.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

HorizonTable = tuple[tuple[float, float], ...]


def elevation_at(table: HorizonTable, azimuth_deg) -> np.ndarray:
    """Skyline elevation (deg) in each given azimuth, linear between samples.

    Empty table -> 0 everywhere (open sky). Wraps around 360, so samples at 350 and
    10 deg blend across North.
    """
    az_query = np.asarray(azimuth_deg, dtype=float) % 360.0
    if not table:
        return np.zeros_like(az_query)
    az = np.array([p[0] for p in table], dtype=float) % 360.0
    el = np.array([p[1] for p in table], dtype=float)
    order = np.argsort(az)
    return np.interp(az_query, az[order], el[order], period=360.0)


def merge(*tables: HorizonTable) -> HorizonTable:
    """Combine horizon tables into one that is the tallest of them in every direction.

    Exact, not resampled: the upper envelope of piecewise-linear skylines only bends
    at an input sample or where two skylines cross, so those are the output samples.
    """
    tables = tuple(t for t in tables if t)
    if not tables:
        return ()
    az = np.unique(np.concatenate([[p[0] % 360.0 for p in t] for t in tables]))
    ring = np.append(az, az[0] + 360.0)
    elev = [elevation_at(t, ring) for t in tables]

    crossings = []
    for a, b in combinations(elev, 2):
        d = a - b
        i = np.nonzero(d[:-1] * d[1:] < 0)[0]
        crossings.append(ring[i] + d[i] / (d[i] - d[i + 1]) * (ring[i + 1] - ring[i]))

    az = np.unique(np.concatenate([az, *crossings]) % 360.0)
    top = np.max([elevation_at(t, az) for t in tables], axis=0)
    return tuple(zip(az.tolist(), top.tolist(), strict=True))


def flat_horizon() -> HorizonTable:
    """Open sky: nothing blocks the sun above the true horizon."""
    return ()


def ridge(azimuth_deg: float, elevation_deg: float, width_deg: float = 30.0) -> HorizonTable:
    """A synthetic terrain block centered on an azimuth.

    Rises to `elevation_deg` at `azimuth_deg`, tapering to 0 at +/- width. Handy for
    tests ("a 30 deg ridge to the east") and for eyeballing a real blocker by hand.
    """
    return (
        ((azimuth_deg - width_deg) % 360, 0.0),
        (azimuth_deg % 360, elevation_deg),
        ((azimuth_deg + width_deg) % 360, 0.0),
    )


def from_elevation_api(*_args, **_kwargs) -> HorizonTable:  # pragma: no cover
    """FUTURE: raytrace a bare-earth skyline from an elevation API (see BACKLOG.md)."""
    raise NotImplementedError("elevation-API horizon raytrace not implemented yet")


def from_lidar_pointcloud(*_args, **_kwargs) -> HorizonTable:  # pragma: no cover
    """FUTURE: opaque horizon from classified rock returns + canopy attenuation.

    See BACKLOG.md. Intentionally a separate, optional sub-project.
    """
    raise NotImplementedError("LiDAR point-cloud horizon not implemented yet")

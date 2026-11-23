"""Horizon tables of (azimuth_deg, elevation_deg) skyline samples, and their sources.

Combine sources with `merge`, never by concatenation. Planned sources: BACKLOG.md.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

HorizonTable = tuple[tuple[float, float], ...]


def elevation_at(table: HorizonTable, azimuth_deg) -> np.ndarray:
    """Skyline elevation (deg) at each azimuth, linear between samples and wrapping at 360."""
    az_query = np.asarray(azimuth_deg, dtype=float) % 360.0
    if not table:
        return np.zeros_like(az_query)
    az = np.array([p[0] for p in table], dtype=float) % 360.0
    el = np.array([p[1] for p in table], dtype=float)
    order = np.argsort(az)
    return np.interp(az_query, az[order], el[order], period=360.0)


def merge(*tables: HorizonTable) -> HorizonTable:
    """Tallest skyline per azimuth across tables.

    Exact: the envelope only bends at input samples and pairwise crossings.
    """
    tables = tuple(t for t in tables if t)
    if not tables:
        return ()
    az = np.unique(np.concatenate([[p[0] % 360.0 for p in t] for t in tables]))
    ring = np.append(az, az[0] + 360.0)
    elev = [elevation_at(t, ring) for t in tables]

    crossings = []
    for a, b in combinations(elev, 2):
        gap = a - b
        i = np.flatnonzero(gap[:-1] * gap[1:] < 0)
        frac = gap[i] / (gap[i] - gap[i + 1])
        crossings.append(ring[i] + frac * (ring[i + 1] - ring[i]))

    az = np.unique(np.concatenate([az, *crossings]) % 360.0)
    top = np.max([elevation_at(t, az) for t in tables], axis=0)
    return tuple(zip(az.tolist(), top.tolist(), strict=True))


def flat_horizon() -> HorizonTable:
    """Open sky: nothing blocks the sun above the true horizon."""
    return ()


def ridge(azimuth_deg: float, elevation_deg: float, width_deg: float = 30.0) -> HorizonTable:
    """Synthetic triangular blocker peaking at azimuth_deg, zero at +/- width_deg."""
    return (
        ((azimuth_deg - width_deg) % 360, 0.0),
        (azimuth_deg % 360, elevation_deg),
        ((azimuth_deg + width_deg) % 360, 0.0),
    )


def from_elevation_api(*_args, **_kwargs) -> HorizonTable:  # pragma: no cover
    """FUTURE: bare-earth skyline raytraced from an elevation API."""
    raise NotImplementedError("elevation-API horizon raytrace not implemented yet")


def from_lidar_pointcloud(*_args, **_kwargs) -> HorizonTable:  # pragma: no cover
    """FUTURE: opaque skyline from classified LiDAR rock returns."""
    raise NotImplementedError("LiDAR point-cloud horizon not implemented yet")

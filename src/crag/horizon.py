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

Because these all return the same table shape, the merge is trivial: concatenate
samples from several sources (DEM skyline + a few hand-entered near-field bumps)
into one `Face.horizon`.
"""

from __future__ import annotations

HorizonTable = tuple[tuple[float, float], ...]


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

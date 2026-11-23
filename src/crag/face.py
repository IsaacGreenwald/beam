"""One rock face, the unit of prediction. A boulder climbed on several sides is several Faces."""

from __future__ import annotations

from enum import Enum
from functools import cache

from pydantic import BaseModel, Field
from timezonefinder import TimezoneFinder


@cache
def _tz_finder() -> TimezoneFinder:
    return TimezoneFinder()


@cache
def timezone_at(lat: float, lon: float) -> str:
    """IANA time zone name (e.g. 'America/Denver') at a coordinate, looked up offline."""
    tz = _tz_finder().timezone_at(lat=lat, lng=lon)
    if tz is None:
        raise ValueError(f"no time zone found at ({lat}, {lon})")
    return tz


class Canopy(str, Enum):
    """Vegetation shading level. Not yet used by the engine (see BACKLOG.md)."""

    open = "open"
    dappled = "dappled"
    heavy = "heavy"


class Face(BaseModel):
    """Where a rock face is and which way it points."""

    id: str = Field(..., description="stable slug, e.g. 'hp40-millipede-SW'")
    name: str = ""

    lat: float = Field(..., ge=-90, le=90, description="degrees North")
    lon: float = Field(..., ge=-180, le=180, description="degrees East")
    altitude: float = Field(0.0, description="metres above sea level")

    aspect: float = Field(
        ...,
        ge=0,
        le=360,
        description="compass bearing the face LOOKS TOWARD (N=0, E=90, S=180, W=270)",
    )
    dip: float = Field(
        90.0,
        ge=0,
        le=180,
        description="tilt of the surface from horizontal: 90=vertical, >90=overhang, <90=slab",
    )

    horizon: tuple[tuple[float, float], ...] = Field(
        default=(),
        description="(azimuth_deg, elevation_deg) skyline samples, built with crag.horizon; "
        "empty = open sky",
    )

    canopy: Canopy = Canopy.open

    model_config = {"frozen": True}

    @property
    def normal_elevation(self) -> float:
        """Elevation (deg) of the outward normal: 0 vertical, + slab, - overhang."""
        return 90.0 - self.dip

    @property
    def tz(self) -> str:
        """The face's local time zone, from its coordinates."""
        return timezone_at(self.lat, self.lon)

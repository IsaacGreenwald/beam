"""The Face: one rock face, the unit of prediction.

A single boulder with problems on opposite sides is *multiple* Faces. For now this
holds only what the geometry engine needs (location + orientation). Horizon, canopy,
and rock_type get added when the steps that use them arrive.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Face(BaseModel):
    """A rock face, described by where it is and which way it points."""

    id: str = Field(..., description="stable slug, e.g. 'hp40-millipede-SW'")
    name: str = ""

    lat: float = Field(..., ge=-90, le=90, description="degrees North")
    lon: float = Field(..., ge=-180, le=180, description="degrees East")
    altitude: float = Field(0.0, description="metres above sea level")

    # Orientation of the face surface.
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

    model_config = {"frozen": True}  # a Face is an immutable description

    @property
    def normal_elevation(self) -> float:
        """Elevation angle (deg above horizontal) of the outward face normal.

        Vertical wall (dip 90) -> 0 (normal is horizontal).
        Slab (dip 60, leaning back) -> +30 (normal points skyward).
        Overhang (dip 120) -> -30 (normal points groundward).
        """
        return 90.0 - self.dip

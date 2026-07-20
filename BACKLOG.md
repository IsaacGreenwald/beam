# Backlog / future work

Deferred ideas with a reserved home in the code, so they can drop in without a
rewrite. Not needed for v1.

## Horizon sources (seam: `crag.horizon`)

The engine consumes a `Face.horizon` table and is agnostic to its source. New
generators plug into `crag/horizon.py`.

- **Elevation-API raytrace** (`from_elevation_api`, stubbed) — march rays from the
  face's lat/lon over a free elevation API, take the max elevation angle per
  azimuth. Captures the macro skyline (ridges, valley walls) remotely, ~1 day, no
  site access. The pragmatic first real source.

- **LiDAR point-cloud horizon** (`from_lidar_pointcloud`, stubbed) — the
  interesting one, deliberately kept as its own optional sub-project (also a strong
  standalone portfolio piece: point-cloud classification).
  - Separate **vegetation vs. rock** using multi-return analysis (trees give
    multiple returns / are penetrable; rock gives a single return / is opaque) or
    the ASPRS classification codes that ship with USGS 3DEP data.
  - Build an **opaque horizon** from the rock/ground returns (a high-res DSM
    resolves boulders a 30 m DEM cannot).
  - Derive a **canopy-density attenuation layer** from the vegetation returns to
    parameterize `Face.canopy` automatically, instead of a hand-typed enum.
  - Caveats: trees in a DSM are frozen/opaque/single-season (still better modeled
    as attenuation than geometry); top-down LiDAR cannot give the face's aspect/dip
    (stays manual); US-only, patchy, heavier data + a processing step.

## Canopy / trees (seam: `Face.canopy`, currently a stub enum)

Model trees NOT as horizon geometry but as a seasonal beam-attenuation factor
(open ~1.0, dappled ~0.5, heavy ~0.2), toggled for deciduous leaf-on/leaf-off.
Prefer open, low-canopy faces for the first area so this stays second-order.

## Calibration ground truth (Louisiana / pre-move)

Rock-surface temperature for `calibrate()` — no site access needed:
- Published rock-temp time series (PANGAEA / paper supplements / author request;
  e.g. the Svalbard CryoGrid rock-wall study, Norway 10-yr monitoring). Force with
  ERA5 at the study coords/dates; report held-out degC.
- Optional cheap reality anchor: IR-thermometer spot-checks on local granite /
  sandstone (cemetery headstones, levee riprap).

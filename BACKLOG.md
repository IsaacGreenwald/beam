# Backlog / future work

Deferred ideas with a reserved home in the code, so they can drop in without a
rewrite. Not needed for v1.

## Horizon sources (seam: `crag.horizon`)

The engine consumes a `Face.horizon` table and is agnostic to its source. New
generators plug into `crag/horizon.py`.

- **Elevation-API raytrace** (`from_elevation_api`, stubbed) - march rays from the
  face's lat/lon over a free elevation API, take the max elevation angle per
  azimuth. Captures the macro skyline (ridges, valley walls) remotely, ~1 day, no
  site access. The pragmatic first real source.

- **LiDAR point-cloud horizon** (`from_lidar_pointcloud`, stubbed) - the
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

Rock-surface temperature for `calibrate()` - no site access needed:
- Published rock-temp time series (PANGAEA / paper supplements / author request;
  e.g. the Svalbard CryoGrid rock-wall study, Norway 10-yr monitoring). Force with
  ERA5 at the study coords/dates; report held-out degC.
- Optional cheap reality anchor: IR-thermometer spot-checks on local granite /
  sandstone (cemetery headstones, levee riprap).

## Audit findings deferred to the temperature phase (2026-11-22)

From the five-lens audit with an Opus overseer.
Each was re-verified with a repro; numbers below are the overseer's.

### Correctness (high)

- **calibrate integrates across observation gaps as if each were one hour.**
  `observations.dropna()` then `weather.loc[obs.index]` hands `integrate` a non-uniform index, and it takes the step from `idx[1] - idx[0]` only.
  A 10-hour NaN gap in noiseless obs turned a true (4e5, 0.55) fit into (2.9e5, 0.26).
  Fix: integrate on the full weather index, fold `obs.notna()` into the scored mask, and add a gap-recovery test.
- **The documented `deep_temp` forecast path returns all NaN.**
  `deep_temperature()` on archive history ends where the forecast starts, so `deep_temp.reindex(idx)` is NaN for every forecast hour, and `band()` drops 191 of 192 rows silently.
  Fix: align with ffill/asof (or accept a scalar) and raise if any hour stays uncovered.
- **Thermal beam forcing is hourly and misaligned with Open-Meteo's radiation convention.**
  Open-Meteo hourly radiation is the mean over the preceding hour, but `beam_factor` is sampled at the stamp and both are lerped, so solar forcing lags the sun ~30 min.
  On a clear 2026-06-21 at lat 40, daily beam energy is -13% for an east face and +10% for a west face.
  Hourly sampling also drops short sun windows entirely: a 37-minute skyline-notch window from `sun_windows` gets zero energy.
  Fix: average minute-resolution `beam_factor` over each (t-1h, t] interval and hold `dni * mean_bf` constant across that interval's substeps; test east/west symmetry and agreement with `sun_windows`.

### Robustness (medium)

- **`integrate` trusts its inputs.**
  One NaN in any forcing column, or a misaligned `beam_factor`/`deep_temp`, propagates to every later hour (one NaN at row 10 made 182 of 192 rows NaN); a one-row input raises a bare IndexError.
  `_trim_edges` keeps interior NaN rows despite its docstring, and `fetch_model` doesn't trim at all.
  Fix: validate length, uniform spacing and NaN-free aligned inputs, naming the column and first bad timestamp.
- **`calibrate` fits silently with zero daylight training rows**, returning the initial absorptivity with `train_rmse=nan`.
  Also: `weather.loc[obs.index]` raises a bare KeyError for obs timestamps missing from weather, and without `deep_temp` the deep node starts from early air temp on short records.

### Tests (medium)

- **The physics core has no numeric pins.**
  Mutants that set the convective coefficient to 0, drop `sky_view` from diffuse, or move the lerp midpoint all pass the full suite.
  Fix: golden `net_flux` components, exact `sky_view_factor(dip=60) == 0.75`, and `integrate` against analytic exponential relaxation.
- **Pipeline wiring is unpinned.**
  Mutants that swap `band()` lo for median, drop `deep_temp` in `forecast()`, drop `fetch_model`'s `models` param, or score held-out RMSE on training rows all pass.
- **Test hygiene.** Three separate weather-frame builders, re-imports inside tests, a misplaced test in `test_forecast.py`, and some narrative comments.

### Physics refinements (low)

- **Deep-rock proxy amplitude.** The 120-day trailing mean keeps ~0.83 of the annual amplitude where conduction at damping depth gives ~0.37; damped in practice by the small `deep_conductance`.
- **Ground-reflected shortwave is missing** (~55 W/m^2 on a vertical wall at albedo 0.2, much more over snow).
- **The 5.7 + 3.8v convective coefficient** is usually quoted as including radiation and assumes near-surface wind, but it's fed 10 m wind alongside an explicit longwave term. Literature provenance not re-verified.

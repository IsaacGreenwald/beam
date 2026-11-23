"""Two-node rock energy balance: a diurnal skin coupled to the air and to deep rock."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STEFAN_BOLTZMANN = 5.670374419e-8  # W/m^2/K^4
EMISSIVITY = 0.95
KELVIN = 273.15
RADIATIVE_COEFF_MAX = 8.0  # W/(m^2 K), 4*eps*sigma*T^3 near 330 K

# k / seasonal damping depth for granite: 2.5 / 3.4
DEEP_CONDUCTANCE = 0.7  # W/(m^2 K)
DEEP_WINDOW_HOURS = 24 * 120


@dataclass(frozen=True)
class ThermalParams:
    thermal_mass: float   # J/(m^2 K), = rho * c * effective_depth
    absorptivity: float   # 0-1, shortwave absorbed fraction
    deep_conductance: float = DEEP_CONDUCTANCE  # W/(m^2 K) to interior rock


def convective_coefficient(wind_ms: float) -> float:
    """Exterior convective heat-transfer coefficient, W/m^2/K."""
    return 5.7 + 3.8 * wind_ms


def sky_temperature(temp_air_c: float, cloud_pct: float) -> float:
    """Effective sky temperature (deg C): Swinbank clear sky, blended toward air by cloud."""
    t_air_k = temp_air_c + KELVIN
    t_clear_k = 0.0552 * t_air_k**1.5
    c = cloud_pct / 100.0
    t_sky4 = (1.0 - c) * t_clear_k**4 + c * t_air_k**4
    return t_sky4**0.25 - KELVIN


def deep_temperature(temp_air: pd.Series, window_hours: int = DEEP_WINDOW_HOURS) -> pd.Series:
    """Deep-rock temperature proxy: trailing mean of air temp.

    A window W lags ~W/2, matching the annual wave's ~58 d lag at damping depth.
    """
    return temp_air.rolling(window_hours, min_periods=1).mean()


def net_flux(
    rock_temp_c: float,
    *,
    dni: float,
    diffuse: float,
    temp_air: float,
    wind: float,
    cloud: float,
    beam_factor: float,
    sky_view: float,
    params: ThermalParams,
    temp_deep: float | None = None,
) -> float:
    """Net flux into the rock surface, W/m^2 (negative = cooling). No temp_deep, no conduction."""
    t_rock_k = rock_temp_c + KELVIN
    t_air_k = temp_air + KELVIN
    t_sky_k = sky_temperature(temp_air, cloud) + KELVIN

    solar = params.absorptivity * (dni * beam_factor + diffuse * sky_view)
    convection = convective_coefficient(wind) * (temp_air - rock_temp_c)
    longwave = EMISSIVITY * STEFAN_BOLTZMANN * (
        sky_view * (t_sky_k**4 - t_rock_k**4)
        + (1.0 - sky_view) * (t_air_k**4 - t_rock_k**4)  # ground hemisphere ~ air temp
    )
    deep = 0.0 if temp_deep is None else params.deep_conductance * (temp_deep - rock_temp_c)
    return solar + convection + longwave + deep


def _lerp(a: float, b: float, frac: float) -> float:
    return a + frac * (b - a)


def _required_substeps(
    thermal_mass: float,
    max_wind: float,
    deep_conductance: float,
    hour_seconds: float,
    floor: int,
) -> int:
    """Sub-steps per hour that keep forward Euler stable for these params."""
    k_max = convective_coefficient(max_wind) + RADIATIVE_COEFF_MAX + deep_conductance
    stable_dt = 0.5 * thermal_mass / k_max  # target sub_dt with margin below the CFL limit
    needed = int(np.ceil(hour_seconds / stable_dt)) if stable_dt > 0 else floor
    return int(min(max(floor, needed), 2000))


def integrate(
    weather: pd.DataFrame,
    beam_factor: pd.Series,
    sky_view: float,
    params: ThermalParams,
    *,
    initial_temp: float | None = None,
    substeps: int = 6,
    deep_temp: pd.Series | None = None,
) -> pd.Series:
    """Forward-Euler energy balance -> hourly rock surface temperature (deg C).

    weather needs dni, diffuse, temp_air, wind, cloud on a uniform hourly index.
    Starts at air temp unless initial_temp; deep_temp overrides the trailing-mean proxy.
    """
    idx = weather.index
    hour_seconds = (idx[1] - idx[0]).total_seconds()
    deep = deep_temperature(weather["temp_air"]) if deep_temp is None else deep_temp.reindex(idx)
    forcing = {
        "dni": weather["dni"].to_numpy(),
        "diffuse": weather["diffuse"].to_numpy(),
        "temp_air": weather["temp_air"].to_numpy(),
        "wind": weather["wind"].to_numpy(),
        "cloud": weather["cloud"].to_numpy(),
        "beam_factor": beam_factor.reindex(idx).to_numpy(),
        "temp_deep": deep.to_numpy(),
    }

    substeps = _required_substeps(
        params.thermal_mass, float(np.nanmax(forcing["wind"])), params.deep_conductance,
        hour_seconds, substeps,
    )
    sub_dt = hour_seconds / substeps

    rock = np.empty(len(idx))
    t = forcing["temp_air"][0] if initial_temp is None else initial_temp
    rock[0] = t
    for i in range(len(idx) - 1):
        for s in range(substeps):
            midpoint = (s + 0.5) / substeps
            at_midpoint = {k: _lerp(v[i], v[i + 1], midpoint) for k, v in forcing.items()}
            flux = net_flux(t, **at_midpoint, sky_view=sky_view, params=params)
            t += flux * sub_dt / params.thermal_mass
        rock[i + 1] = t
    return pd.Series(rock, index=idx, name="rock_temp")

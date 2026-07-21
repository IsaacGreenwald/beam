"""Thermal model: hourly surface energy balance -> rock temperature.

Single-bucket lumped model. Two fittable params (thermal_mass, absorptivity);
everything else is fixed from literature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STEFAN_BOLTZMANN = 5.670374419e-8  # W/m^2/K^4
EMISSIVITY = 0.95
KELVIN = 273.15


@dataclass(frozen=True)
class ThermalParams:
    thermal_mass: float   # J/(m^2 K), = rho * c * effective_depth
    absorptivity: float   # 0-1, shortwave absorbed fraction


def convective_coefficient(wind_ms: float) -> float:
    """Exterior convective heat-transfer coefficient, W/m^2/K."""
    return 5.7 + 3.8 * wind_ms


def sky_temperature(temp_air_c: float, cloud_pct: float) -> float:
    """Effective radiative sky temperature (deg C).

    Clear sky (Swinbank) is well below air temp; cloud pulls it back toward air.
    """
    t_air_k = temp_air_c + KELVIN
    t_clear_k = 0.0552 * t_air_k**1.5
    c = cloud_pct / 100.0
    t_sky4 = (1.0 - c) * t_clear_k**4 + c * t_air_k**4
    return t_sky4**0.25 - KELVIN


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
) -> float:
    """Net energy flux INTO the rock surface, W/m^2 (negative = cooling)."""
    t_rock_k = rock_temp_c + KELVIN
    t_air_k = temp_air + KELVIN
    t_sky_k = sky_temperature(temp_air, cloud) + KELVIN

    solar = params.absorptivity * (dni * beam_factor + diffuse * sky_view)
    convection = convective_coefficient(wind) * (temp_air - rock_temp_c)
    longwave = EMISSIVITY * STEFAN_BOLTZMANN * (
        sky_view * (t_sky_k**4 - t_rock_k**4)
        + (1.0 - sky_view) * (t_air_k**4 - t_rock_k**4)  # ground hemisphere ~ air temp
    )
    return solar + convection + longwave


def _required_substeps(thermal_mass: float, max_wind: float, hour_seconds: float, floor: int) -> int:
    """Sub-steps keeping forward-Euler well inside its stability limit for any params."""
    k_max = (5.7 + 3.8 * max_wind) + 8.0  # convective(max wind) + radiative upper bound
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
) -> pd.Series:
    """Step the energy balance forward -> hourly rock surface temperature (deg C).

    weather has columns dni, diffuse, temp_air, wind, cloud on a uniform hourly index;
    beam_factor is aligned to it. Forcing is interpolated across sub-steps; sub-step
    count auto-scales up from `substeps` to stay numerically stable for any thermal_mass.
    Starts at air temperature unless initial_temp given.
    """
    idx = weather.index
    hour_seconds = (idx[1] - idx[0]).total_seconds()
    bf = beam_factor.reindex(idx).to_numpy()
    dni = weather["dni"].to_numpy()
    diffuse = weather["diffuse"].to_numpy()
    temp_air = weather["temp_air"].to_numpy()
    wind = weather["wind"].to_numpy()
    cloud = weather["cloud"].to_numpy()

    substeps = _required_substeps(params.thermal_mass, float(np.nanmax(wind)), hour_seconds, substeps)
    sub_dt = hour_seconds / substeps

    rock = np.empty(len(idx))
    t = temp_air[0] if initial_temp is None else initial_temp
    rock[0] = t
    for i in range(len(idx) - 1):
        for s in range(substeps):
            f = (s + 0.5) / substeps  # interpolate forcing to the sub-step midpoint
            flux = net_flux(
                t,
                dni=dni[i] + f * (dni[i + 1] - dni[i]),
                diffuse=diffuse[i] + f * (diffuse[i + 1] - diffuse[i]),
                temp_air=temp_air[i] + f * (temp_air[i + 1] - temp_air[i]),
                wind=wind[i] + f * (wind[i + 1] - wind[i]),
                cloud=cloud[i] + f * (cloud[i + 1] - cloud[i]),
                beam_factor=bf[i] + f * (bf[i + 1] - bf[i]),
                sky_view=sky_view,
                params=params,
            )
            t += flux * sub_dt / params.thermal_mass
        rock[i + 1] = t
    return pd.Series(rock, index=idx, name="rock_temp")

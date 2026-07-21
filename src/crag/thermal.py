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
    beam_factor is aligned to it. Starts at air temperature unless initial_temp given.
    """
    dt = (weather.index[1] - weather.index[0]).total_seconds() / substeps
    bf = beam_factor.reindex(weather.index).to_numpy()
    dni = weather["dni"].to_numpy()
    diffuse = weather["diffuse"].to_numpy()
    temp_air = weather["temp_air"].to_numpy()
    wind = weather["wind"].to_numpy()
    cloud = weather["cloud"].to_numpy()

    rock = np.empty(len(weather))
    t = temp_air[0] if initial_temp is None else initial_temp
    rock[0] = t
    for i in range(len(weather) - 1):
        for _ in range(substeps):
            flux = net_flux(
                t, dni=dni[i], diffuse=diffuse[i], temp_air=temp_air[i],
                wind=wind[i], cloud=cloud[i], beam_factor=bf[i],
                sky_view=sky_view, params=params,
            )
            t += flux * dt / params.thermal_mass
        rock[i + 1] = t
    return pd.Series(rock, index=weather.index, name="rock_temp")

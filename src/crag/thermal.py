"""Thermal model: hourly surface energy balance -> rock temperature.

Single-bucket lumped model. Two fittable params (thermal_mass, absorptivity);
everything else is fixed from literature.
"""

from __future__ import annotations

from dataclasses import dataclass

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

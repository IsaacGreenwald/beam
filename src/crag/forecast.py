"""Combine geometry + weather + thermal into per-model rock-temp forecasts."""

from __future__ import annotations

import pandas as pd

from crag.face import Face
from crag.geometry import beam_factor, sky_view_factor
from crag.thermal import ThermalParams, integrate
from crag.weather import with_dew_point


def forecast_model(face: Face, weather: pd.DataFrame, params: ThermalParams) -> pd.DataFrame:
    """One model's weather -> temp_air, rock_temp, dew_point, margin."""
    w = with_dew_point(weather)
    rock = integrate(w, beam_factor(face, w.index), sky_view_factor(face), params)
    return pd.DataFrame(
        {
            "temp_air": w["temp_air"],
            "rock_temp": rock,
            "dew_point": w["dew_point"],
            "margin": rock - w["dew_point"],
        }
    )


def forecast(
    face: Face,
    weather_tables: dict[str, pd.DataFrame],
    params: ThermalParams,
) -> dict[str, pd.DataFrame]:
    """{model: weather} -> {model: forecast}. The spread across models is the band."""
    return {m: forecast_model(face, w, params) for m, w in weather_tables.items()}


def band(results: dict[str, pd.DataFrame], column: str = "rock_temp") -> pd.DataFrame:
    """Across-model lo/median/hi per timestep: ATMOSPHERIC uncertainty only.

    This is model disagreement, distinct from calibration/held-out error; the two
    must never be merged into a single band.
    """
    frame = pd.DataFrame({m: df[column] for m, df in results.items()}).dropna()
    return pd.DataFrame(
        {"lo": frame.min(axis=1), "median": frame.median(axis=1), "hi": frame.max(axis=1)}
    )

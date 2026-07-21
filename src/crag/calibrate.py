"""Calibration: fit the 2 thermal params to observed rock temperature.

Source-agnostic seam: observations are a tz-aware Series of observed rock temp
(deg C). The fit does not care whether they came from a dataset, reanalysis, or a
logger. Reports error only on a held-out window the fit never saw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from crag.face import Face
from crag.geometry import beam_factor, sky_view_factor, sun_position
from crag.thermal import ThermalParams, integrate

DAYLIGHT_WEIGHT = 3.0
DEFAULT_BOUNDS = ((1.0e4, 0.1), (5.0e6, 1.0))  # (thermal_mass, absorptivity) lo/hi


@dataclass(frozen=True)
class CalibrationResult:
    params: ThermalParams
    held_out_rmse: float  # daylight-hours RMSE on the held-out window, deg C
    train_rmse: float     # daylight-hours RMSE on the training window, deg C
    n_train: int
    n_held_out: int


def _daylight_mask(face: Face, index: pd.DatetimeIndex) -> np.ndarray:
    return sun_position(face, index)["elevation"].to_numpy() > 0.0


def _daylight_rmse(pred: np.ndarray, obs: np.ndarray, is_day: np.ndarray) -> float:
    err = (pred - obs)[is_day]
    return float(np.sqrt(np.mean(err**2)))


def calibrate(
    face: Face,
    weather: pd.DataFrame,
    observations: pd.Series,
    *,
    train_frac: float = 0.7,
    initial: ThermalParams = ThermalParams(3.0e5, 0.6),
    bounds: tuple[tuple[float, float], tuple[float, float]] = DEFAULT_BOUNDS,
) -> CalibrationResult:
    """Fit (thermal_mass, absorptivity) on the training window; score on held-out."""
    obs = observations.dropna()
    weather = weather.loc[obs.index]
    idx = obs.index
    observed = obs.to_numpy()

    # geometry is param-independent -> compute once, reuse every iteration
    bf = beam_factor(face, idx)
    sky_view = sky_view_factor(face)
    is_day = _daylight_mask(face, idx)
    sqrt_w = np.sqrt(np.where(is_day, DAYLIGHT_WEIGHT, 1.0))

    n_train = int(np.floor(len(idx) * train_frac))
    train, held = slice(0, n_train), slice(n_train, len(idx))

    def predict(x: np.ndarray) -> np.ndarray:
        return integrate(weather, bf, sky_view, ThermalParams(x[0], x[1])).to_numpy()

    def residuals(x: np.ndarray) -> np.ndarray:
        return (sqrt_w * (predict(x) - observed))[train]

    lo, hi = bounds
    fit = least_squares(
        residuals,
        x0=[initial.thermal_mass, initial.absorptivity],
        bounds=([lo[0], lo[1]], [hi[0], hi[1]]),
    )
    params = ThermalParams(float(fit.x[0]), float(fit.x[1]))

    pred = predict(fit.x)
    return CalibrationResult(
        params=params,
        held_out_rmse=_daylight_rmse(pred[held], observed[held], is_day[held]),
        train_rmse=_daylight_rmse(pred[train], observed[train], is_day[train]),
        n_train=n_train,
        n_held_out=len(idx) - n_train,
    )

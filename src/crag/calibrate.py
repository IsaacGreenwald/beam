"""Fit thermal params to observed rock temperature; score on a held-out window."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from crag.face import Face
from crag.geometry import beam_factor, sky_view_factor, sun_position
from crag.thermal import ThermalParams, integrate

DAYLIGHT_WEIGHT = 3.0
# lo/hi for (thermal_mass, absorptivity, deep_conductance)
DEFAULT_INITIAL = ThermalParams(3.0e5, 0.6)
DEFAULT_BOUNDS = ((1.0e4, 0.1, 0.0), (5.0e6, 1.0, 5.0))


@dataclass(frozen=True)
class CalibrationResult:
    params: ThermalParams
    held_out_rmse: float  # daylight-hours RMSE on the held-out window, deg C
    train_rmse: float     # daylight-hours RMSE on the training window, deg C
    n_train: int          # scored training rows
    n_held_out: int       # scored held-out rows


def warm_season_mask(
    weather: pd.DataFrame, min_daily_mean: float = 2.0, window_hours: int = 24
) -> pd.Series:
    """Hours whose centred daily mean air temp is above freezing (excludes freeze-thaw)."""
    daily = weather["temp_air"].rolling(window_hours, center=True, min_periods=1).mean()
    return daily > min_daily_mean


def _daylight_mask(face: Face, index: pd.DatetimeIndex) -> np.ndarray:
    return sun_position(face, index)["elevation"].to_numpy() > 0.0


def _rmse(pred: np.ndarray, obs: np.ndarray, rows: np.ndarray) -> float:
    if not rows.any():
        return float("nan")
    return float(np.sqrt(np.mean(((pred - obs)[rows]) ** 2)))


def calibrate(
    face: Face,
    weather: pd.DataFrame,
    observations: pd.Series,
    *,
    train_frac: float = 0.7,
    initial: ThermalParams = DEFAULT_INITIAL,
    bounds: tuple[tuple[float, ...], tuple[float, ...]] = DEFAULT_BOUNDS,
    mask: pd.Series | None = None,
    fit_deep: bool = False,
    deep_temp: pd.Series | None = None,
) -> CalibrationResult:
    """Fit on the first train_frac of observations; report daylight RMSE on the rest.

    mask limits scoring, not integration. fit_deep frees deep_conductance.
    """
    obs = observations.dropna()
    weather = weather.loc[obs.index]
    idx = obs.index
    observed = obs.to_numpy()

    bf = beam_factor(face, idx)
    sky_view = sky_view_factor(face)
    is_day = _daylight_mask(face, idx)
    sqrt_w = np.sqrt(np.where(is_day, DAYLIGHT_WEIGHT, 1.0))

    scored = np.ones(len(idx), bool) if mask is None else mask.reindex(idx).fillna(False).to_numpy()
    is_train = np.zeros(len(idx), bool)
    is_train[: int(np.floor(len(idx) * train_frac))] = True
    fit_rows, held_rows = scored & is_train, scored & ~is_train
    if not fit_rows.any():
        raise ValueError("mask leaves no training rows to fit on")

    n_free = 3 if fit_deep else 2

    def predict(x: np.ndarray) -> np.ndarray:
        params = ThermalParams(x[0], x[1], x[2] if fit_deep else initial.deep_conductance)
        return integrate(weather, bf, sky_view, params, deep_temp=deep_temp).to_numpy()

    def residuals(x: np.ndarray) -> np.ndarray:
        return (sqrt_w * (predict(x) - observed))[fit_rows]

    lo, hi = bounds
    x0 = [initial.thermal_mass, initial.absorptivity, initial.deep_conductance][:n_free]
    fit = least_squares(residuals, x0=x0, bounds=(list(lo[:n_free]), list(hi[:n_free])))

    pred = predict(fit.x)
    return CalibrationResult(
        params=ThermalParams(
            float(fit.x[0]),
            float(fit.x[1]),
            float(fit.x[2]) if fit_deep else initial.deep_conductance,
        ),
        held_out_rmse=_rmse(pred, observed, held_rows & is_day),
        train_rmse=_rmse(pred, observed, fit_rows & is_day),
        n_train=int((fit_rows & is_day).sum()),
        n_held_out=int((held_rows & is_day).sum()),
    )

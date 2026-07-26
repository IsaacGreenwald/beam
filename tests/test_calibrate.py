import numpy as np
import pandas as pd
import pytest

from crag.calibrate import calibrate, warm_season_mask
from crag.face import Face
from crag.geometry import beam_factor, sky_view_factor
from crag.thermal import ThermalParams, integrate

FACE = Face(id="cal-S", lat=40.0, lon=-105.0, aspect=180, dip=90)
TRUE = ThermalParams(thermal_mass=4.0e5, absorptivity=0.55)


def _synthetic_weather(days=10):
    idx = pd.date_range("2026-07-01", periods=days * 24, freq="h", tz="UTC")
    hour = idx.hour.to_numpy()
    return pd.DataFrame(
        {
            "dni": np.full(len(idx), 900.0),      # gated by beam_factor
            "diffuse": np.full(len(idx), 100.0),
            "temp_air": 15.0 + 8.0 * np.sin(2 * np.pi * (hour - 9) / 24),
            "rh": np.full(len(idx), 40.0),
            "wind": np.full(len(idx), 2.0),
            "cloud": np.full(len(idx), 10.0),
        },
        index=idx,
    )


def _observed(weather, params, noise=0.0, seed=0, deep_temp=None):
    bf = beam_factor(FACE, weather.index)
    rock = integrate(weather, bf, sky_view_factor(FACE), params, deep_temp=deep_temp)
    if noise:
        rock = rock + np.random.default_rng(seed).normal(0.0, noise, len(rock))
    return rock.rename("observed")


def test_recovers_known_params_noiseless():
    w = _synthetic_weather()
    obs = _observed(w, TRUE)
    result = calibrate(FACE, w, obs)

    assert np.isclose(result.params.thermal_mass, TRUE.thermal_mass, rtol=0.05)
    assert np.isclose(result.params.absorptivity, TRUE.absorptivity, atol=0.03)
    assert result.held_out_rmse < 0.1  # essentially exact on noiseless data


def test_held_out_reported_separately_and_reasonable_under_noise():
    w = _synthetic_weather()
    obs = _observed(w, TRUE, noise=0.5, seed=1)
    result = calibrate(FACE, w, obs)

    # params still close to truth despite noise
    assert np.isclose(result.params.thermal_mass, TRUE.thermal_mass, rtol=0.25)
    assert np.isclose(result.params.absorptivity, TRUE.absorptivity, atol=0.1)
    # held-out error is a real, separate number in the noise ballpark
    assert 0.0 < result.held_out_rmse < 2.0
    assert result.n_train > result.n_held_out  # 70/30 split


def test_warm_season_mask_excludes_the_freezing_spell():
    w = _synthetic_weather(days=20)
    cold = w.index < w.index[0] + pd.Timedelta(days=10)
    w.loc[cold, "temp_air"] = -12.0
    m = warm_season_mask(w)
    assert not m[cold].any()      # winter half dropped entirely
    assert m[~cold].iloc[12:].all()  # summer kept past the centred window's feathered edge


def test_mask_restricts_scoring_but_not_the_fit_window():
    w = _synthetic_weather(days=20)
    obs = _observed(w, TRUE)
    m = pd.Series(w.index >= w.index[0] + pd.Timedelta(days=5), index=w.index)

    full = calibrate(FACE, w, obs)
    masked = calibrate(FACE, w, obs, mask=m)
    # fewer scored rows, but the integration still spans the whole record
    assert masked.n_train + masked.n_held_out < full.n_train + full.n_held_out
    assert np.isclose(masked.params.thermal_mass, TRUE.thermal_mass, rtol=0.05)
    assert masked.held_out_rmse < 0.1


def test_mask_with_no_training_rows_raises():
    w = _synthetic_weather(days=4)
    obs = _observed(w, TRUE)
    none_early = pd.Series(w.index >= w.index[-1], index=w.index)
    with pytest.raises(ValueError, match="no training rows"):
        calibrate(FACE, w, obs, mask=none_early)


def test_fit_deep_recovers_conductance_when_interior_departs_from_air():
    # deep node is identifiable only when T_deep is far from air temp
    w = _synthetic_weather(days=20)
    deep = pd.Series(np.full(len(w), -15.0), index=w.index)
    truth = ThermalParams(4.0e5, 0.55, deep_conductance=1.2)
    obs = _observed(w, truth, deep_temp=deep)

    result = calibrate(FACE, w, obs, fit_deep=True, deep_temp=deep)
    assert np.isclose(result.params.deep_conductance, 1.2, atol=0.15)
    assert np.isclose(result.params.thermal_mass, truth.thermal_mass, rtol=0.1)
    assert result.held_out_rmse < 0.2


def test_deep_conductance_untouched_when_not_fitted():
    w = _synthetic_weather()
    obs = _observed(w, TRUE)
    result = calibrate(FACE, w, obs, initial=ThermalParams(3.0e5, 0.6, 0.9))
    assert result.params.deep_conductance == 0.9  # held fixed, not optimized

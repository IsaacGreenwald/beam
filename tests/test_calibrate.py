import numpy as np
import pandas as pd

from crag.calibrate import calibrate
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


def _observed(weather, params, noise=0.0, seed=0):
    bf = beam_factor(FACE, weather.index)
    rock = integrate(weather, bf, sky_view_factor(FACE), params)
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

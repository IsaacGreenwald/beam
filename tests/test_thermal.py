import numpy as np
import pandas as pd

from crag.thermal import ThermalParams, integrate, net_flux, sky_temperature

PARAMS = ThermalParams(thermal_mass=2.0e5, absorptivity=0.6)

# a vertical open face: sees half the sky
SKY_VIEW = 0.5


def _night_flux(rock_temp_c, cloud, temp_air=10.0):
    return net_flux(
        rock_temp_c,
        dni=0.0,
        diffuse=0.0,
        temp_air=temp_air,
        wind=1.0,
        cloud=cloud,
        beam_factor=0.0,
        sky_view=SKY_VIEW,
        params=PARAMS,
    )


def test_sky_is_colder_than_air_when_clear():
    assert sky_temperature(15.0, cloud_pct=0) < 15.0
    # full cloud pulls the sky back up toward air temperature
    assert sky_temperature(15.0, cloud_pct=100) > sky_temperature(15.0, cloud_pct=0)


def test_sun_adds_energy():
    day = net_flux(
        20.0, dni=800.0, diffuse=120.0, temp_air=20.0, wind=1.0, cloud=0.0,
        beam_factor=0.9, sky_view=SKY_VIEW, params=PARAMS,
    )
    shade = net_flux(
        20.0, dni=800.0, diffuse=120.0, temp_air=20.0, wind=1.0, cloud=0.0,
        beam_factor=0.0, sky_view=SKY_VIEW, params=PARAMS,
    )
    assert day > shade


def test_clear_night_cools_more_than_cloudy():
    # rock warmer than air at night -> both lose heat, but clear sky loses more
    clear = _night_flux(rock_temp_c=12.0, cloud=0.0)
    cloudy = _night_flux(rock_temp_c=12.0, cloud=100.0)
    assert clear < cloudy < 0


def test_convection_pulls_toward_air():
    hot_rock = _night_flux(rock_temp_c=25.0, cloud=100.0, temp_air=10.0)
    cold_rock = _night_flux(rock_temp_c=-5.0, cloud=100.0, temp_air=10.0)
    assert hot_rock < 0  # hot rock in cool air -> loses heat
    assert cold_rock > 0  # cold rock in warmer air -> gains heat


def _weather(hours, *, temp_air, dni=0.0, diffuse=0.0, wind=1.0, cloud=100.0):
    idx = pd.date_range("2026-07-20", periods=hours, freq="h", tz="UTC")
    n = len(idx)
    frame = pd.DataFrame(
        {
            "dni": np.full(n, dni) if np.isscalar(dni) else dni,
            "diffuse": np.full(n, diffuse),
            "temp_air": temp_air if hasattr(temp_air, "__len__") else np.full(n, temp_air),
            "wind": np.full(n, wind),
            "cloud": np.full(n, cloud),
        },
        index=idx,
    )
    return frame, pd.Series(np.zeros(n), index=idx)  # beam_factor all zero (no sun)


def test_integrate_length_and_index():
    w, bf = _weather(24, temp_air=15.0)
    rock = integrate(w, bf, sky_view=0.5, params=PARAMS)
    assert len(rock) == 24
    assert rock.index.equals(w.index)


def test_relaxes_toward_air_when_no_sun():
    # warm rock, cool air, cloudy (minimal radiation): rock should cool monotonically
    w, bf = _weather(24, temp_air=10.0, cloud=100.0)
    rock = integrate(w, bf, sky_view=0.5, params=PARAMS, initial_temp=25.0)
    assert rock.iloc[-1] < rock.iloc[0]
    assert np.all(np.diff(rock.to_numpy()) < 0)  # strictly cooling
    assert rock.iloc[-1] > 10.0  # but not overshooting past air temp


def test_larger_thermal_mass_changes_slower():
    w, bf = _weather(24, temp_air=10.0, cloud=100.0)
    light = integrate(w, bf, 0.5, ThermalParams(5.0e4, 0.6), initial_temp=25.0)
    heavy = integrate(w, bf, 0.5, ThermalParams(1.0e6, 0.6), initial_temp=25.0)
    # heavier rock stays closer to its starting temperature
    assert abs(heavy.iloc[-1] - 25.0) < abs(light.iloc[-1] - 25.0)


def test_rock_temp_lags_air_temp():
    # 3 days of sinusoidal air temp, no sun; rock is dragged by air but lags it
    hours = 72
    idx = np.arange(hours)
    air = 15.0 + 10.0 * np.sin(2 * np.pi * (idx - 6) / 24)  # peak at hour 12 each day
    w, bf = _weather(hours, temp_air=air, cloud=100.0)
    rock = integrate(w, bf, sky_view=0.5, params=ThermalParams(4.0e5, 0.6)).to_numpy()

    # compare on day 3 to avoid startup transient
    day3 = slice(48, 72)
    air_peak = 48 + np.argmax(air[day3])
    rock_peak = 48 + np.argmax(rock[day3])
    assert rock_peak > air_peak  # rock peaks after air

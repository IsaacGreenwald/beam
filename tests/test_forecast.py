import numpy as np
import pandas as pd

from crag.face import Face
from crag.forecast import band, forecast, forecast_model
from crag.geometry import sky_view_factor
from crag.thermal import ThermalParams

PARAMS = ThermalParams(thermal_mass=3.0e5, absorptivity=0.6)


def _weather(hours=48, temp_air=20.0, dni=600.0):
    idx = pd.date_range("2026-07-20", periods=hours, freq="h", tz="UTC")
    n = len(idx)
    return pd.DataFrame(
        {
            "dni": np.full(n, dni),
            "diffuse": np.full(n, 100.0),
            "temp_air": np.full(n, temp_air),
            "rh": np.full(n, 50.0),
            "wind": np.full(n, 1.5),
            "cloud": np.full(n, 20.0),
        },
        index=idx,
    )


def test_sky_view_factor_by_dip():
    assert sky_view_factor(Face(id="v", lat=40, lon=-105, aspect=180, dip=90)) == 0.5
    slab = sky_view_factor(Face(id="s", lat=40, lon=-105, aspect=180, dip=60))
    overhang = sky_view_factor(Face(id="o", lat=40, lon=-105, aspect=180, dip=120))
    assert slab > 0.5 > overhang  # slab sees more sky, overhang less


def test_forecast_model_columns_and_margin():
    face = Face(id="S", lat=40, lon=-105, aspect=180, dip=90)
    out = forecast_model(face, _weather(), PARAMS)
    assert list(out.columns) == ["temp_air", "rock_temp", "dew_point", "margin"]
    # margin is exactly rock_temp - dew_point
    assert np.allclose(out["margin"], out["rock_temp"] - out["dew_point"])


def test_forecast_runs_per_model():
    face = Face(id="S", lat=40, lon=-105, aspect=180, dip=90)
    tables = {"ecmwf": _weather(dni=600.0), "gfs": _weather(dni=900.0)}
    results = forecast(face, tables, PARAMS)
    assert set(results) == {"ecmwf", "gfs"}
    # more direct sun -> hotter rock
    assert results["gfs"]["rock_temp"].max() > results["ecmwf"]["rock_temp"].max()


def test_band_bounds_and_alignment():
    face = Face(id="S", lat=40, lon=-105, aspect=180, dip=90)
    tables = {"ecmwf": _weather(dni=600.0), "gfs": _weather(dni=900.0)}
    b = band(forecast(face, tables, PARAMS))
    assert list(b.columns) == ["lo", "median", "hi"]
    assert (b["lo"] <= b["median"]).all()
    assert (b["median"] <= b["hi"]).all()
    assert not b.isna().any().any()  # aligned on shared timestamps

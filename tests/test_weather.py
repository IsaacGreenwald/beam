import pandas as pd

from crag.weather import FIELDS, parse_hourly


def test_parse_hourly_shapes_and_utc():
    hourly = {
        "time": ["2026-07-20T00:00", "2026-07-20T01:00", "2026-07-20T02:00"],
        "direct_normal_irradiance": [0.0, 0.0, 120.0],
        "diffuse_radiation": [0.0, 5.0, 40.0],
        "temperature_2m": [18.0, 17.5, 19.0],
        "relative_humidity_2m": [60, 62, 55],
        "wind_speed_10m": [1.2, 1.0, 2.3],
        "cloud_cover": [10, 0, 25],
    }
    df = parse_hourly(hourly)

    assert list(df.columns) == list(FIELDS.values())
    assert len(df) == 3
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df["dni"].iloc[2] == 120.0

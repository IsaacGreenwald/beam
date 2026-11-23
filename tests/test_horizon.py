"""Horizon tables: interpolation and merging multiple skyline sources."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from crag.horizon import elevation_at, flat_horizon, merge, ridge

DENSE = np.arange(0.0, 360.0, 0.1)

ridges = st.builds(
    ridge,
    azimuth_deg=st.floats(0, 360),
    elevation_deg=st.floats(0, 80),
    width_deg=st.floats(1, 90),
)


def _envelope(*tables):
    return np.max([elevation_at(t, DENSE) for t in tables], axis=0)


def test_overlapping_ridges_keep_the_taller_one():
    a, b = ridge(90, 40, 45), ridge(120, 20, 30)
    merged = merge(a, b)
    assert np.isclose(elevation_at(merged, 100.0), 31.11, atol=0.01)
    assert np.isclose(elevation_at(merged, 130.0), 13.33, atol=0.01)


def test_merge_with_flat_is_identity():
    a = ridge(200, 25, 40)
    assert np.allclose(elevation_at(merge(a, flat_horizon()), DENSE), elevation_at(a, DENSE))


def test_merge_with_self_is_identity():
    a = ridge(10, 30, 50)
    assert np.allclose(elevation_at(merge(a, a), DENSE), elevation_at(a, DENSE))


def test_merge_of_nothing_is_open_sky():
    assert merge() == ()
    assert merge(flat_horizon(), flat_horizon()) == ()


def test_crossing_across_north_is_exact():
    a, b = ridge(350, 30, 40), ridge(20, 30, 40)
    assert np.allclose(elevation_at(merge(a, b), DENSE), _envelope(a, b))


@given(st.lists(ridges, min_size=1, max_size=4))
def test_merge_is_exact_upper_envelope(tables):
    assert np.allclose(elevation_at(merge(*tables), DENSE), _envelope(*tables), atol=1e-6)

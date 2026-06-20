"""Tests for the registration layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from he2ihc_align.registration.registrar import HEIHCRegistrar
from he2ihc_align.slide_io.factory import open_slide

TEST_DATA = Path("/home/fengyifan/disk/code/valis/test_SCCE")


class MockRegistrar:
    """Deterministic mock registrar for fast unit tests."""

    def __init__(self, offset_x: float = 5.0, offset_y: float = 10.0):
        self.offset_x = offset_x
        self.offset_y = offset_y

    def warp_xy_from_he_to_ihc(self, xy: np.ndarray, marker: str) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)
        return xy + np.array([self.offset_x, self.offset_y])

    def warp_xy_from_ihc_to_he(self, xy: np.ndarray, marker: str) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)
        return xy - np.array([self.offset_x, self.offset_y])


def test_mock_registrar_warp_xy_from_he_to_ihc():
    """Fast unit test with a mock registrar."""
    registrar = MockRegistrar(offset_x=5.0, offset_y=10.0)
    he_corners = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    ihc_corners = registrar.warp_xy_from_he_to_ihc(he_corners, marker="CD3")
    assert ihc_corners.shape == (4, 2)
    expected = he_corners + np.array([5.0, 10.0])
    np.testing.assert_array_almost_equal(ihc_corners, expected)


def test_mock_registrar_warp_xy_from_ihc_to_he():
    """Fast unit test for inverse warp."""
    registrar = MockRegistrar(offset_x=5.0, offset_y=10.0)
    ihc_corners = np.array([[5.0, 10.0], [105.0, 10.0], [105.0, 110.0], [5.0, 110.0]])
    he_corners = registrar.warp_xy_from_ihc_to_he(ihc_corners, marker="CD3")
    assert he_corners.shape == (4, 2)
    expected = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    np.testing.assert_array_almost_equal(he_corners, expected)


@pytest.mark.slow
def test_registrar_fits_and_warps_corners():
    he_path = next(TEST_DATA.glob("174162-1/174162-1-第一批/*.kfb"))
    ihc_paths = sorted((TEST_DATA / "174162-1/174162-1-第一批").glob("*.svs"))[:2]

    he_slide = open_slide(he_path)
    ihc_slides = {p.stem.split()[-1]: open_slide(p) for p in ihc_paths}

    registrar = HEIHCRegistrar(
        he_slide=he_slide,
        ihc_slides=ihc_slides,
        registration_level=3,
    )
    registrar.fit()

    he_corners = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    for marker in ihc_slides:
        ihc_corners = registrar.warp_xy_from_he_to_ihc(he_corners, marker=marker)
        assert ihc_corners.shape == (4, 2)

"""Tests for the registration layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from he2ihc_align.slide_io.factory import open_slide
from he2ihc_align.registration.registrar import HEIHCRegistrar


TEST_DATA = Path("/home/fengyifan/disk/code/valis/test_SCCE")


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

"""Tests for slide_io backends, factory, and case discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from he2ihc_align.slide_io.base import Slide
from he2ihc_align.slide_io.openslide_backend import OpenSlideBackend
from he2ihc_align.slide_io.kfb_backend import KfbSlideBackend
from he2ihc_align.slide_io.factory import open_slide
from he2ihc_align.case_io import discover_case


TEST_DATA = Path("/home/fengyifan/disk/code/valis/test_SCCE")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def svs_path():
    case_dir = TEST_DATA / "174162-1" / "174162-1-第一批"
    return case_dir / "174162-1 CD3.svs"


@pytest.fixture(scope="session")
def kfb_path():
    case_dir = TEST_DATA / "174162-1" / "174162-1-第一批"
    return case_dir / "174162-1.kfb"


@pytest.fixture
def openslide_backend(svs_path):
    with OpenSlideBackend(svs_path) as slide:
        yield slide


@pytest.fixture
def kfb_backend(kfb_path):
    with KfbSlideBackend(kfb_path) as slide:
        yield slide


# ---------------------------------------------------------------------------
# OpenSlideBackend tests
# ---------------------------------------------------------------------------

class TestOpenSlideBackend:
    def test_is_slide_instance(self, openslide_backend):
        assert isinstance(openslide_backend, Slide)

    def test_level_count_positive(self, openslide_backend):
        assert openslide_backend.level_count > 0

    def test_level_dimensions_match_count(self, openslide_backend):
        assert len(openslide_backend.level_dimensions) == openslide_backend.level_count

    def test_level_downsamples_match_count(self, openslide_backend):
        assert len(openslide_backend.level_downsamples) == openslide_backend.level_count

    def test_properties_is_dict(self, openslide_backend):
        assert isinstance(openslide_backend.properties, dict)

    def test_read_region_shape_and_dtype(self, openslide_backend):
        arr = openslide_backend.read_region((0, 0), 0, (256, 256))
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (256, 256, 3)
        assert arr.dtype == np.uint8

    def test_read_region_different_level(self, openslide_backend):
        if openslide_backend.level_count > 1:
            arr = openslide_backend.read_region((0, 0), 1, (128, 128))
            assert arr.shape == (128, 128, 3)
            assert arr.dtype == np.uint8

    def test_get_best_level_for_downsample(self, openslide_backend):
        level = openslide_backend.get_best_level_for_downsample(4.0)
        assert 0 <= level < openslide_backend.level_count


# ---------------------------------------------------------------------------
# KfbSlideBackend tests
# ---------------------------------------------------------------------------

class TestKfbSlideBackend:
    def test_is_slide_instance(self, kfb_backend):
        assert isinstance(kfb_backend, Slide)

    def test_level_count_positive(self, kfb_backend):
        assert kfb_backend.level_count > 0

    def test_level_dimensions_match_count(self, kfb_backend):
        assert len(kfb_backend.level_dimensions) == kfb_backend.level_count

    def test_level_downsamples_match_count(self, kfb_backend):
        assert len(kfb_backend.level_downsamples) == kfb_backend.level_count

    def test_properties_is_dict(self, kfb_backend):
        assert isinstance(kfb_backend.properties, dict)

    def test_read_region_shape_and_dtype(self, kfb_backend):
        arr = kfb_backend.read_region((0, 0), 0, (256, 256))
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (256, 256, 3)
        assert arr.dtype == np.uint8

    def test_read_region_different_level(self, kfb_backend):
        if kfb_backend.level_count > 1:
            arr = kfb_backend.read_region((0, 0), 1, (128, 128))
            assert arr.shape == (128, 128, 3)
            assert arr.dtype == np.uint8

    def test_get_best_level_for_downsample(self, kfb_backend):
        level = kfb_backend.get_best_level_for_downsample(4.0)
        assert 0 <= level < kfb_backend.level_count


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestFactory:
    def test_open_svs_returns_openslide(self, svs_path):
        slide = open_slide(svs_path)
        assert isinstance(slide, OpenSlideBackend)
        slide.close()

    def test_open_kfb_returns_kfb(self, kfb_path):
        slide = open_slide(kfb_path)
        assert isinstance(slide, KfbSlideBackend)
        slide.close()

    def test_open_unsupported_raises(self, tmp_path):
        bad = tmp_path / "foo.txt"
        bad.write_text("not a slide")
        with pytest.raises(ValueError, match="Unsupported"):
            open_slide(bad)


# ---------------------------------------------------------------------------
# Case discovery tests
# ---------------------------------------------------------------------------

class TestDiscoverCase:
    def test_discover_174162_1(self):
        case_dir = TEST_DATA / "174162-1"
        he_path, markers = discover_case(case_dir)
        assert he_path.suffix.lower() == ".kfb"
        assert he_path.stem == "174162-1"
        assert isinstance(markers, dict)
        assert len(markers) > 0
        # Check marker names are last space-separated token of stem
        for marker_name, path in markers.items():
            assert path.suffix.lower() == ".svs"
            expected = path.stem.split()[-1]
            assert marker_name == expected

    def test_discover_raises_when_no_kfb(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            discover_case(tmp_path)

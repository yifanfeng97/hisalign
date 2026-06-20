"""Tests for slide_io backends, factory, and case discovery."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from hisalign.case_io import discover_case
from hisalign.slide_io.base import Slide, SlideIOError
from hisalign.slide_io.factory import open_slide
from hisalign.slide_io.image_backend import ImageSlideBackend
from hisalign.slide_io.kfb_backend import KfbSlideBackend
from hisalign.slide_io.openslide_backend import OpenSlideBackend

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
# MockSlide fixture for fast unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_slide():
    """Return a MagicMock that satisfies the Slide protocol."""
    slide = MagicMock()
    slide.level_count = 4
    slide.level_dimensions = [(10000, 10000), (5000, 5000), (2500, 2500), (1250, 1250)]
    slide.level_downsamples = [1.0, 2.0, 4.0, 8.0]
    slide.properties = {"mock": "true"}

    def _read_region(location, level, size):
        return np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)

    slide.read_region = _read_region
    slide.get_best_level_for_downsample = lambda d: 0
    return slide


# ---------------------------------------------------------------------------
# OpenSlideBackend tests
# ---------------------------------------------------------------------------


class TestOpenSlideBackend:
    @pytest.mark.slow
    def test_is_slide_instance(self, openslide_backend):
        assert isinstance(openslide_backend, Slide)

    @pytest.mark.slow
    def test_level_count_positive(self, openslide_backend):
        assert openslide_backend.level_count > 0

    @pytest.mark.slow
    def test_level_dimensions_match_count(self, openslide_backend):
        assert len(openslide_backend.level_dimensions) == openslide_backend.level_count

    @pytest.mark.slow
    def test_level_downsamples_match_count(self, openslide_backend):
        assert len(openslide_backend.level_downsamples) == openslide_backend.level_count

    @pytest.mark.slow
    def test_properties_is_dict(self, openslide_backend):
        assert isinstance(openslide_backend.properties, dict)

    @pytest.mark.slow
    def test_read_region_shape_and_dtype(self, openslide_backend):
        arr = openslide_backend.read_region((0, 0), 0, (256, 256))
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (256, 256, 3)
        assert arr.dtype == np.uint8

    @pytest.mark.slow
    def test_read_region_different_level(self, openslide_backend):
        if openslide_backend.level_count <= 1:
            pytest.skip("Only one level available")
        arr = openslide_backend.read_region((0, 0), 1, (128, 128))
        assert arr.shape == (128, 128, 3)
        assert arr.dtype == np.uint8

    @pytest.mark.slow
    def test_read_region_level0_location_at_higher_level(self, openslide_backend):
        """Verify that location is interpreted as level-0 coordinates."""
        if openslide_backend.level_count <= 3:
            pytest.skip("Need at least 4 levels")
        level = 3
        downsample = openslide_backend.level_downsamples[level]
        # Pick a small patch at level-0 coordinates that is well within bounds
        level0_w, level0_h = openslide_backend.level_dimensions[0]
        x0 = min(1000, int(level0_w * 0.1))
        y0 = min(1000, int(level0_h * 0.1))
        size = (64, 64)
        # Ensure the patch fits in level-0
        assert x0 + size[0] * downsample <= level0_w
        assert y0 + size[1] * downsample <= level0_h
        arr = openslide_backend.read_region((x0, y0), level, size)
        assert arr.shape == (64, 64, 3)
        assert arr.dtype == np.uint8

    @pytest.mark.slow
    def test_read_region_invalid_level_raises(self, openslide_backend):
        with pytest.raises(SlideIOError):
            openslide_backend.read_region(
                (0, 0), openslide_backend.level_count + 1, (128, 128)
            )

    @pytest.mark.slow
    def test_read_region_out_of_bounds_raises(self, openslide_backend):
        with pytest.raises(SlideIOError):
            openslide_backend.read_region((999999999, 999999999), 0, (256, 256))

    @pytest.mark.slow
    def test_get_best_level_for_downsample(self, openslide_backend):
        level = openslide_backend.get_best_level_for_downsample(4.0)
        assert 0 <= level < openslide_backend.level_count


# ---------------------------------------------------------------------------
# KfbSlideBackend tests
# ---------------------------------------------------------------------------


class TestKfbSlideBackend:
    @pytest.mark.slow
    def test_is_slide_instance(self, kfb_backend):
        assert isinstance(kfb_backend, Slide)

    @pytest.mark.slow
    def test_level_count_positive(self, kfb_backend):
        assert kfb_backend.level_count > 0

    @pytest.mark.slow
    def test_level_dimensions_match_count(self, kfb_backend):
        assert len(kfb_backend.level_dimensions) == kfb_backend.level_count

    @pytest.mark.slow
    def test_level_downsamples_match_count(self, kfb_backend):
        assert len(kfb_backend.level_downsamples) == kfb_backend.level_count

    @pytest.mark.slow
    def test_properties_is_dict(self, kfb_backend):
        assert isinstance(kfb_backend.properties, dict)

    @pytest.mark.slow
    def test_read_region_shape_and_dtype(self, kfb_backend):
        arr = kfb_backend.read_region((0, 0), 0, (256, 256))
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (256, 256, 3)
        assert arr.dtype == np.uint8

    @pytest.mark.slow
    def test_read_region_different_level(self, kfb_backend):
        if kfb_backend.level_count <= 1:
            pytest.skip("Only one level available")
        arr = kfb_backend.read_region((0, 0), 1, (128, 128))
        assert arr.shape == (128, 128, 3)
        assert arr.dtype == np.uint8

    @pytest.mark.slow
    def test_read_region_level0_location_at_higher_level(self, kfb_backend):
        """Verify that location is interpreted as level-0 coordinates."""
        if kfb_backend.level_count <= 3:
            pytest.skip("Need at least 4 levels")
        level = 3
        downsample = kfb_backend.level_downsamples[level]
        # Pick a small patch at level-0 coordinates that is well within bounds
        level0_w, level0_h = kfb_backend.level_dimensions[0]
        x0 = min(1000, int(level0_w * 0.1))
        y0 = min(1000, int(level0_h * 0.1))
        size = (64, 64)
        # Ensure the patch fits in level-0
        assert x0 + size[0] * downsample <= level0_w
        assert y0 + size[1] * downsample <= level0_h
        arr = kfb_backend.read_region((x0, y0), level, size)
        assert arr.shape == (64, 64, 3)
        assert arr.dtype == np.uint8

    @pytest.mark.slow
    def test_read_region_invalid_level_raises(self, kfb_backend):
        with pytest.raises(SlideIOError):
            kfb_backend.read_region((0, 0), kfb_backend.level_count + 1, (128, 128))

    @pytest.mark.slow
    def test_read_region_out_of_bounds_raises(self, kfb_backend):
        with pytest.raises(SlideIOError):
            kfb_backend.read_region((999999999, 999999999), 0, (256, 256))

    @pytest.mark.slow
    def test_get_best_level_for_downsample(self, kfb_backend):
        level = kfb_backend.get_best_level_for_downsample(4.0)
        assert 0 <= level < kfb_backend.level_count


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFactory:
    @pytest.mark.slow
    def test_open_svs_returns_openslide(self, svs_path):
        slide = open_slide(svs_path)
        assert isinstance(slide, OpenSlideBackend)
        slide.close()

    @pytest.mark.slow
    def test_open_kfb_returns_kfb(self, kfb_path):
        slide = open_slide(kfb_path)
        assert isinstance(slide, KfbSlideBackend)
        slide.close()

    def test_open_unsupported_raises(self, tmp_path):
        bad = tmp_path / "foo.txt"
        bad.write_text("not a slide")
        with pytest.raises(ValueError, match="Unsupported"):
            open_slide(bad)


class TestImageSlideBackend:
    """Fast unit tests for the static-image backend."""

    @pytest.fixture
    def rgb_path(self, tmp_path):
        img = np.zeros((120, 200, 3), dtype=np.uint8)
        img[:, :100] = [255, 0, 0]
        img[:, 100:] = [0, 255, 0]
        path = tmp_path / "test.png"
        from PIL import Image

        Image.fromarray(img).save(path)
        return path

    def test_open_png_returns_image_backend(self, rgb_path):
        slide = open_slide(rgb_path)
        assert isinstance(slide, ImageSlideBackend)
        slide.close()

    def test_level_dimensions_match_image_size(self, rgb_path):
        with ImageSlideBackend(rgb_path) as slide:
            assert slide.level_count == 1
            assert slide.level_dimensions == [(200, 120)]
            assert slide.level_downsamples == [1.0]

    def test_read_region_returns_hwc_uint8(self, rgb_path):
        with ImageSlideBackend(rgb_path) as slide:
            arr = slide.read_region((50, 30), 0, (80, 60))
            assert arr.shape == (60, 80, 3)
            assert arr.dtype == np.uint8

    def test_read_region_clamps_to_bounds(self, rgb_path):
        with ImageSlideBackend(rgb_path) as slide:
            arr = slide.read_region((150, 80), 0, (100, 100))
            assert arr.shape == (40, 50, 3)

    def test_read_region_invalid_level_raises(self, rgb_path):
        with ImageSlideBackend(rgb_path) as slide:
            with pytest.raises(SlideIOError):
                slide.read_region((0, 0), 1, (10, 10))

    def test_get_best_level_returns_zero(self, rgb_path):
        with ImageSlideBackend(rgb_path) as slide:
            assert slide.get_best_level_for_downsample(4.0) == 0


# ---------------------------------------------------------------------------
# Case discovery tests
# ---------------------------------------------------------------------------


class TestDiscoverCase:
    @pytest.mark.slow
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

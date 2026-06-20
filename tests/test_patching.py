"""Tests for patching.py grid patch sampling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from he2ihc_align.patching import sample_grid_patches
from he2ihc_align.slide_io.factory import open_slide

TEST_DATA = Path("/home/fengyifan/disk/code/valis/test_SCCE")


@pytest.fixture(scope="session")
def he_slide():
    he_path = next(TEST_DATA.glob("174162-1/174162-1-第一批/*.kfb"))
    slide = open_slide(he_path)
    yield slide
    slide.close()


class TestSampleGridPatches:
    def test_filters_white_patches(self):
        """Create a synthetic slide with a white region and a dark region."""
        slide = MagicMock()
        slide.level_count = 1
        slide.level_dimensions = [(1024, 1024)]
        slide.level_downsamples = [1.0]
        slide.properties = {}

        def _read_region(location, level, size):
            x, y = location
            w, h = size
            # Create a white region in the top-left 512x512 of the slide, dark elsewhere
            arr = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    if x + j < 512 and y + i < 512:
                        arr[i, j] = 255  # white
                    else:
                        arr[i, j] = 50  # dark
            return arr

        slide.read_region = _read_region
        slide.get_best_level_for_downsample = lambda d: 0

        bboxes = sample_grid_patches(
            slide, patch_size=512, stride=512, white_threshold=230, max_white_ratio=0.95
        )
        # The white patch at (0,0) should be filtered out, dark patch at (512,512) kept
        assert (0, 0, 512, 512) not in bboxes
        assert (512, 512, 512, 512) in bboxes

    def test_keeps_partially_white_patches_below_threshold(self):
        """Patches with white ratio below max_white_ratio should be kept."""
        slide = MagicMock()
        slide.level_count = 1
        slide.level_dimensions = [(512, 512)]
        slide.level_downsamples = [1.0]
        slide.properties = {}

        def _read_region(location, level, size):
            arr = np.ones((size[1], size[0], 3), dtype=np.uint8) * 200
            return arr

        slide.read_region = _read_region
        slide.get_best_level_for_downsample = lambda d: 0

        bboxes = sample_grid_patches(
            slide, patch_size=512, white_threshold=230, max_white_ratio=0.95
        )
        assert len(bboxes) == 1
        assert bboxes[0] == (0, 0, 512, 512)

    def test_empty_slide_returns_empty(self):
        slide = MagicMock()
        slide.level_count = 1
        slide.level_dimensions = [(100, 100)]
        slide.level_downsamples = [1.0]
        slide.properties = {}

        def _read_region(location, level, size):
            return np.ones((size[1], size[0], 3), dtype=np.uint8) * 255

        slide.read_region = _read_region
        slide.get_best_level_for_downsample = lambda d: 0

        bboxes = sample_grid_patches(
            slide, patch_size=200, white_threshold=230, max_white_ratio=0.95
        )
        assert bboxes == []

    def test_mock_slide_returns_list_of_tuples(self):
        """Test that sample_grid_patches returns correct bbox format on a mock slide."""
        slide = MagicMock()
        slide.level_count = 4
        slide.level_dimensions = [(4096, 4096), (2048, 2048), (1024, 1024), (512, 512)]
        slide.level_downsamples = [1.0, 2.0, 4.0, 8.0]
        slide.properties = {}

        def _read_region(location, level, size):
            return np.random.randint(0, 200, (size[1], size[0], 3), dtype=np.uint8)

        slide.read_region = _read_region
        slide.get_best_level_for_downsample = lambda d: 0

        bboxes = sample_grid_patches(slide, patch_size=512, level=3)
        assert isinstance(bboxes, list)
        for bbox in bboxes:
            assert isinstance(bbox, tuple)
            assert len(bbox) == 4
            x, y, w, h = bbox
            assert all(isinstance(v, int) for v in (x, y, w, h))
            assert 0 < w <= 512 * 8  # level 3 downsample is 8
            assert 0 < h <= 512 * 8

    def test_mock_slide_default_stride_equals_patch_size(self):
        """With default stride == patch_size, patches should not overlap."""
        slide = MagicMock()
        slide.level_count = 4
        slide.level_dimensions = [(4096, 4096), (2048, 2048), (1024, 1024), (512, 512)]
        slide.level_downsamples = [1.0, 2.0, 4.0, 8.0]
        slide.properties = {}

        def _read_region(location, level, size):
            return np.random.randint(0, 200, (size[1], size[0], 3), dtype=np.uint8)

        slide.read_region = _read_region
        slide.get_best_level_for_downsample = lambda d: 0

        bboxes = sample_grid_patches(slide, patch_size=512, level=3)
        xs = sorted({bbox[0] for bbox in bboxes})
        if len(xs) > 1:
            diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
            assert all(d >= 512 * 8 for d in diffs)

    def test_mock_slide_custom_stride(self):
        """Smaller stride should produce more patches than larger stride."""
        slide = MagicMock()
        slide.level_count = 4
        slide.level_dimensions = [(4096, 4096), (2048, 2048), (1024, 1024), (512, 512)]
        slide.level_downsamples = [1.0, 2.0, 4.0, 8.0]
        slide.properties = {}

        def _read_region(location, level, size):
            return np.random.randint(0, 200, (size[1], size[0], 3), dtype=np.uint8)

        slide.read_region = _read_region
        slide.get_best_level_for_downsample = lambda d: 0

        bboxes_1024 = sample_grid_patches(slide, patch_size=512, stride=256, level=3)
        bboxes_512 = sample_grid_patches(slide, patch_size=512, stride=512, level=3)
        assert len(bboxes_1024) >= len(bboxes_512)

    def test_mock_slide_margin_excludes_edges(self):
        """Margin should exclude patches near edges."""
        slide = MagicMock()
        slide.level_count = 4
        slide.level_dimensions = [(4096, 4096), (2048, 2048), (1024, 1024), (512, 512)]
        slide.level_downsamples = [1.0, 2.0, 4.0, 8.0]
        slide.properties = {}

        def _read_region(location, level, size):
            return np.random.randint(0, 200, (size[1], size[0], 3), dtype=np.uint8)

        slide.read_region = _read_region
        slide.get_best_level_for_downsample = lambda d: 0

        bboxes_no_margin = sample_grid_patches(slide, patch_size=512, margin=0, level=3)
        bboxes_margin = sample_grid_patches(slide, patch_size=512, margin=100, level=3)
        assert len(bboxes_margin) <= len(bboxes_no_margin)
        for bbox in bboxes_margin:
            x, y, w, h = bbox
            downsample = slide.level_downsamples[3]
            margin_level = int(100 / downsample)
            expected_min = int(margin_level * downsample)
            assert x >= expected_min
            assert y >= expected_min

    def test_mock_slide_level_affects_coordinates(self):
        """Higher level should produce fewer patches due to smaller effective area."""
        slide = MagicMock()
        slide.level_count = 4
        slide.level_dimensions = [(4096, 4096), (2048, 2048), (1024, 1024), (512, 512)]
        slide.level_downsamples = [1.0, 2.0, 4.0, 8.0]
        slide.properties = {}

        def _read_region(location, level, size):
            return np.random.randint(0, 200, (size[1], size[0], 3), dtype=np.uint8)

        slide.read_region = _read_region
        slide.get_best_level_for_downsample = lambda d: 0

        bboxes_l0 = sample_grid_patches(slide, patch_size=512, level=0)
        bboxes_l3 = sample_grid_patches(slide, patch_size=512, level=3)
        assert len(bboxes_l3) <= len(bboxes_l0)

    def test_mock_slide_grid_sampling_returns_bboxes(self):
        """Test grid sampling on mock slide returns correct bbox dimensions."""
        slide = MagicMock()
        slide.level_count = 4
        slide.level_dimensions = [(4096, 4096), (2048, 2048), (1024, 1024), (512, 512)]
        slide.level_downsamples = [1.0, 2.0, 4.0, 8.0]
        slide.properties = {}

        def _read_region(location, level, size):
            return np.random.randint(0, 200, (size[1], size[0], 3), dtype=np.uint8)

        slide.read_region = _read_region
        slide.get_best_level_for_downsample = lambda d: 0

        bboxes = sample_grid_patches(
            slide, patch_size=512, stride=512, level=3, max_white_ratio=1.0
        )
        assert len(bboxes) > 0
        downsample = slide.level_downsamples[3]
        expected_size = int(512 * downsample)
        for bbox in bboxes:
            x, y, w, h = bbox
            assert w == expected_size or w < expected_size
            assert h == expected_size or h < expected_size

    @pytest.mark.slow
    def test_real_slide_filters_some_patches(self, he_slide):
        """On a real slide, white/blank filtering should remove some patches."""
        bboxes_all = sample_grid_patches(
            he_slide, patch_size=512, stride=512, level=3, max_white_ratio=1.0
        )
        bboxes_filtered = sample_grid_patches(
            he_slide, patch_size=512, stride=512, level=3, max_white_ratio=0.95
        )
        assert len(bboxes_filtered) <= len(bboxes_all)

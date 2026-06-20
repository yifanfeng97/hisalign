"""Tests for mapping.py HE-to-IHC patch mapping."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from he2ihc_align.mapping import build_mapping_table
from he2ihc_align.registration.registrar import HEIHCRegistrar
from he2ihc_align.slide_io.factory import open_slide


TEST_DATA = Path("/home/fengyifan/disk/code/valis/test_SCCE")


class MockRegistrar:
    """Deterministic mock registrar for fast unit tests."""

    def __init__(self, offset_x: float = 10.0, offset_y: float = 20.0):
        self.offset_x = offset_x
        self.offset_y = offset_y

    def warp_xy_from_he_to_ihc(self, xy: np.ndarray, marker: str) -> np.ndarray:
        """Return corners shifted by a fixed offset."""
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)
        return xy + np.array([self.offset_x, self.offset_y])


@pytest.fixture
def mock_registrar():
    return MockRegistrar(offset_x=10.0, offset_y=20.0)


@pytest.fixture
def mock_he_slide():
    slide = MagicMock()
    slide.level_dimensions = [(1000, 1000)]
    return slide


@pytest.fixture
def mock_ihc_slides():
    slides = {
        "CD3": MagicMock(),
        "CD4": MagicMock(),
    }
    for slide in slides.values():
        slide.level_dimensions = [(1000, 1000)]
    return slides


@pytest.fixture(scope="session")
def real_registrar():
    he_path = next(TEST_DATA.glob("174162-1/174162-1-第一批/*.kfb"))
    ihc_paths = sorted((TEST_DATA / "174162-1/174162-1-第一批").glob("*.svs"))[:2]

    he_slide = open_slide(he_path)
    ihc_slides = {p.stem.split()[-1]: open_slide(p) for p in ihc_paths}

    reg = HEIHCRegistrar(
        he_slide=he_slide,
        ihc_slides=ihc_slides,
        registration_level=3,
    )
    reg.fit()

    yield reg

    he_slide.close()
    for s in ihc_slides.values():
        s.close()


@pytest.fixture(scope="session")
def real_he_slide():
    he_path = next(TEST_DATA.glob("174162-1/174162-1-第一批/*.kfb"))
    slide = open_slide(he_path)
    yield slide
    slide.close()


@pytest.fixture(scope="session")
def real_ihc_slides():
    ihc_paths = sorted((TEST_DATA / "174162-1/174162-1-第一批").glob("*.svs"))[:2]
    slides = {p.stem.split()[-1]: open_slide(p) for p in ihc_paths}
    yield slides
    for s in slides.values():
        s.close()


class TestBuildMappingTable:
    def test_returns_dataframe(self, mock_registrar, mock_he_slide, mock_ihc_slides):
        he_patch_bboxes = [(0, 0, 512, 512), (512, 512, 512, 512)]
        df = build_mapping_table(
            registrar=mock_registrar,
            he_slide=mock_he_slide,
            ihc_slides=mock_ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        assert isinstance(df, pd.DataFrame)

    def test_outputs_expected_columns(self, mock_registrar, mock_he_slide, mock_ihc_slides):
        """Test that build_mapping_table outputs the expected columns."""
        he_patch_bboxes = [(0, 0, 512, 512), (512, 512, 512, 512), (256, 256, 512, 512)]
        df = build_mapping_table(
            registrar=mock_registrar,
            he_slide=mock_he_slide,
            ihc_slides=mock_ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        expected_cols = [
            "patch_id",
            "slide_id",
            "marker",
            "he_x",
            "he_y",
            "he_w",
            "he_h",
            "ihc_x",
            "ihc_y",
            "ihc_w",
            "ihc_h",
            "clipped",
        ]
        assert list(df.columns) == expected_cols
        assert len(df) == 3 * len(mock_ihc_slides)

    def test_row_count_matches_patches_times_markers(self, mock_registrar, mock_he_slide, mock_ihc_slides):
        he_patch_bboxes = [(0, 0, 512, 512), (512, 512, 512, 512)]
        df = build_mapping_table(
            registrar=mock_registrar,
            he_slide=mock_he_slide,
            ihc_slides=mock_ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        expected_rows = len(he_patch_bboxes) * len(mock_ihc_slides)
        assert len(df) == expected_rows

    def test_patch_id_format(self, mock_registrar, mock_he_slide, mock_ihc_slides):
        he_patch_bboxes = [(0, 0, 512, 512)]
        df = build_mapping_table(
            registrar=mock_registrar,
            he_slide=mock_he_slide,
            ihc_slides=mock_ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        for patch_id in df["patch_id"]:
            assert isinstance(patch_id, str)
            assert patch_id.startswith("test_slide_")

    def test_he_coords_match_input(self, mock_registrar, mock_he_slide, mock_ihc_slides):
        he_patch_bboxes = [(100, 200, 512, 512)]
        df = build_mapping_table(
            registrar=mock_registrar,
            he_slide=mock_he_slide,
            ihc_slides=mock_ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        row = df[df["marker"] == list(mock_ihc_slides.keys())[0]].iloc[0]
        assert row["he_x"] == 100
        assert row["he_y"] == 200
        assert row["he_w"] == 512
        assert row["he_h"] == 512

    def test_ihc_bbox_positive(self, mock_registrar, mock_he_slide, mock_ihc_slides):
        he_patch_bboxes = [(0, 0, 512, 512)]
        df = build_mapping_table(
            registrar=mock_registrar,
            he_slide=mock_he_slide,
            ihc_slides=mock_ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        for _, row in df.iterrows():
            assert row["ihc_w"] > 0
            assert row["ihc_h"] > 0

    def test_clipped_is_boolean(self, mock_registrar, mock_he_slide, mock_ihc_slides):
        he_patch_bboxes = [(0, 0, 512, 512)]
        df = build_mapping_table(
            registrar=mock_registrar,
            he_slide=mock_he_slide,
            ihc_slides=mock_ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        assert df["clipped"].dtype == bool or set(df["clipped"].unique()).issubset({True, False})

    def test_clipped_true_when_out_of_bounds(self, mock_registrar, mock_he_slide, mock_ihc_slides):
        """If the mapped IHC bbox exceeds slide dimensions, clipped should be True."""
        # Use a very large HE coordinate that will map outside IHC
        mock_he_slide.level_dimensions = [(500, 500)]
        for slide in mock_ihc_slides.values():
            slide.level_dimensions = [(500, 500)]
        he_patch_bboxes = [(400, 400, 512, 512)]
        df = build_mapping_table(
            registrar=mock_registrar,
            he_slide=mock_he_slide,
            ihc_slides=mock_ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        # At least some rows should be clipped for edge patches
        assert df["clipped"].any(), "Expected at least one clipped row for edge patch"

    def test_slide_id_in_all_rows(self, mock_registrar, mock_he_slide, mock_ihc_slides):
        he_patch_bboxes = [(0, 0, 512, 512)]
        df = build_mapping_table(
            registrar=mock_registrar,
            he_slide=mock_he_slide,
            ihc_slides=mock_ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="my_slide_123",
        )
        assert (df["slide_id"] == "my_slide_123").all()

    def test_unique_patch_id_per_marker(self, mock_registrar, mock_he_slide, mock_ihc_slides):
        he_patch_bboxes = [(0, 0, 512, 512), (512, 512, 512, 512)]
        df = build_mapping_table(
            registrar=mock_registrar,
            he_slide=mock_he_slide,
            ihc_slides=mock_ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        # Each patch_id should appear once per marker
        for marker in df["marker"].unique():
            marker_df = df[df["marker"] == marker]
            assert marker_df["patch_id"].nunique() == len(marker_df)

    def test_ihc_coords_have_fixed_offset(self, mock_registrar, mock_he_slide, mock_ihc_slides):
        """MockRegistrar adds fixed offset; verify IHC coords reflect that."""
        he_patch_bboxes = [(0, 0, 100, 100)]
        df = build_mapping_table(
            registrar=mock_registrar,
            he_slide=mock_he_slide,
            ihc_slides=mock_ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        row = df[df["marker"] == "CD3"].iloc[0]
        # HE corners: (0,0), (100,0), (100,100), (0,100)
        # After offset +10, +20: (10,20), (110,20), (110,120), (10,120)
        # Enclosing bbox: x=10, y=20, w=100, h=100
        assert row["ihc_x"] == 10
        assert row["ihc_y"] == 20
        assert row["ihc_w"] == 100
        assert row["ihc_h"] == 100

    @pytest.mark.slow
    def test_build_mapping_table_with_real_registrar(self, real_registrar, real_he_slide, real_ihc_slides):
        """Integration test with real registrar and slides."""
        from he2ihc_align.patching import sample_grid_patches
        he_patch_bboxes = sample_grid_patches(
            real_he_slide, patch_size=512, stride=512, level=3, max_white_ratio=1.0
        )[:3]
        assert len(he_patch_bboxes) == 3

        df = build_mapping_table(
            registrar=real_registrar,
            he_slide=real_he_slide,
            ihc_slides=real_ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="174162-1",
        )
        expected_cols = [
            "patch_id",
            "slide_id",
            "marker",
            "he_x",
            "he_y",
            "he_w",
            "he_h",
            "ihc_x",
            "ihc_y",
            "ihc_w",
            "ihc_h",
            "clipped",
        ]
        assert list(df.columns) == expected_cols
        assert len(df) == 3 * len(real_ihc_slides)

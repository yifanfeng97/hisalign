"""Tests for mapping.py HE-to-IHC patch mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from he2ihc_align.mapping import build_mapping_table
from he2ihc_align.registration.registrar import HEIHCRegistrar
from he2ihc_align.slide_io.factory import open_slide


TEST_DATA = Path("/home/fengyifan/disk/code/valis/test_SCCE")


@pytest.fixture(scope="session")
def registrar():
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
def he_slide():
    he_path = next(TEST_DATA.glob("174162-1/174162-1-第一批/*.kfb"))
    slide = open_slide(he_path)
    yield slide
    slide.close()


@pytest.fixture(scope="session")
def ihc_slides():
    ihc_paths = sorted((TEST_DATA / "174162-1/174162-1-第一批").glob("*.svs"))[:2]
    slides = {p.stem.split()[-1]: open_slide(p) for p in ihc_paths}
    yield slides
    for s in slides.values():
        s.close()


class TestBuildMappingTable:
    def test_returns_dataframe(self, registrar, he_slide, ihc_slides):
        he_patch_bboxes = [(0, 0, 512, 512), (512, 512, 512, 512)]
        df = build_mapping_table(
            registrar=registrar,
            he_slide=he_slide,
            ihc_slides=ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        assert isinstance(df, pd.DataFrame)

    def test_build_mapping_table_outputs_expected_columns(self, registrar, he_slide, ihc_slides):
        """Test from requirements: discover case, fit registrar, sample 3 patches, build table."""
        # Sample 3 grid patches from HE at level 3
        from he2ihc_align.patching import sample_grid_patches
        he_patch_bboxes = sample_grid_patches(
            he_slide, patch_size=512, stride=512, level=3, max_white_ratio=1.0
        )[:3]
        assert len(he_patch_bboxes) == 3

        df = build_mapping_table(
            registrar=registrar,
            he_slide=he_slide,
            ihc_slides=ihc_slides,
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
        assert len(df) == 3 * len(ihc_slides)

    def test_row_count_matches_patches_times_markers(self, registrar, he_slide, ihc_slides):
        he_patch_bboxes = [(0, 0, 512, 512), (512, 512, 512, 512)]
        df = build_mapping_table(
            registrar=registrar,
            he_slide=he_slide,
            ihc_slides=ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        expected_rows = len(he_patch_bboxes) * len(ihc_slides)
        assert len(df) == expected_rows

    def test_patch_id_format(self, registrar, he_slide, ihc_slides):
        he_patch_bboxes = [(0, 0, 512, 512)]
        df = build_mapping_table(
            registrar=registrar,
            he_slide=he_slide,
            ihc_slides=ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        for patch_id in df["patch_id"]:
            assert isinstance(patch_id, str)
            assert patch_id.startswith("test_slide_")

    def test_he_coords_match_input(self, registrar, he_slide, ihc_slides):
        he_patch_bboxes = [(100, 200, 512, 512)]
        df = build_mapping_table(
            registrar=registrar,
            he_slide=he_slide,
            ihc_slides=ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        row = df[df["marker"] == list(ihc_slides.keys())[0]].iloc[0]
        assert row["he_x"] == 100
        assert row["he_y"] == 200
        assert row["he_w"] == 512
        assert row["he_h"] == 512

    def test_ihc_bbox_positive(self, registrar, he_slide, ihc_slides):
        he_patch_bboxes = [(0, 0, 512, 512)]
        df = build_mapping_table(
            registrar=registrar,
            he_slide=he_slide,
            ihc_slides=ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        for _, row in df.iterrows():
            assert row["ihc_w"] > 0
            assert row["ihc_h"] > 0

    def test_clipped_is_boolean(self, registrar, he_slide, ihc_slides):
        he_patch_bboxes = [(0, 0, 512, 512)]
        df = build_mapping_table(
            registrar=registrar,
            he_slide=he_slide,
            ihc_slides=ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        assert df["clipped"].dtype == bool or set(df["clipped"].unique()).issubset({True, False})

    def test_clipped_true_when_out_of_bounds(self, registrar, he_slide, ihc_slides):
        """If the mapped IHC bbox exceeds slide dimensions, clipped should be True."""
        # Use a very large HE coordinate that will likely map outside IHC
        w, h = he_slide.level_dimensions[0]
        he_patch_bboxes = [(w - 512, h - 512, 512, 512)]
        df = build_mapping_table(
            registrar=registrar,
            he_slide=he_slide,
            ihc_slides=ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        # At least some rows should be clipped for edge patches
        assert df["clipped"].any(), "Expected at least one clipped row for edge patch"

    def test_slide_id_in_all_rows(self, registrar, he_slide, ihc_slides):
        he_patch_bboxes = [(0, 0, 512, 512)]
        df = build_mapping_table(
            registrar=registrar,
            he_slide=he_slide,
            ihc_slides=ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="my_slide_123",
        )
        assert (df["slide_id"] == "my_slide_123").all()

    def test_unique_patch_id_per_marker(self, registrar, he_slide, ihc_slides):
        he_patch_bboxes = [(0, 0, 512, 512), (512, 512, 512, 512)]
        df = build_mapping_table(
            registrar=registrar,
            he_slide=he_slide,
            ihc_slides=ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id="test_slide",
        )
        # Each patch_id should appear once per marker
        for marker in df["marker"].unique():
            marker_df = df[df["marker"] == marker]
            assert marker_df["patch_id"].nunique() == len(marker_df)

"""Tests for CLI run_case and main entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from he2ihc_align.cli import main, run_case


TEST_DATA = Path("/home/fengyifan/disk/code/valis/test_SCCE")


class MockRegistrar:
    """Deterministic mock registrar for fast CLI tests."""

    def __init__(self, offset_x: float = 10.0, offset_y: float = 20.0):
        self.offset_x = offset_x
        self.offset_y = offset_y

    def fit(self):
        return self

    def warp_xy_from_he_to_ihc(self, xy, marker):
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)
        return xy + np.array([self.offset_x, self.offset_y])


class TestRunCaseMocked:
    """Fast unit tests that mock all heavy dependencies."""

    @patch("he2ihc_align.cli.open_slide")
    @patch("he2ihc_align.cli.HEIHCRegistrar")
    @patch("he2ihc_align.cli.sample_grid_patches")
    @patch("he2ihc_align.cli.build_mapping_table")
    @patch("he2ihc_align.cli.create_html_gallery")
    @patch("he2ihc_align.cli.read_patch_rgb")
    @patch("he2ihc_align.cli.discover_case")
    def test_run_case_produces_csv(
        self,
        mock_discover_case,
        mock_read_patch_rgb,
        mock_create_html_gallery,
        mock_build_mapping_table,
        mock_sample_grid_patches,
        mock_HEIHCRegistrar,
        mock_open_slide,
        tmp_path,
    ):
        """Test that run_case orchestrates components and writes a CSV."""
        # Setup mocks - create directory structure: parent/case_dir/...
        parent_dir = tmp_path / "test_case"
        parent_dir.mkdir()
        case_dir = parent_dir / "batch"
        case_dir.mkdir()
        he_path = case_dir / "test.kfb"
        he_path.write_text("mock")
        markers = {"CD3": case_dir / "test CD3.svs"}
        for p in markers.values():
            p.write_text("mock")
        mock_discover_case.return_value = (he_path, markers)

        he_slide = MagicMock()
        ihc_slide = MagicMock()
        mock_open_slide.side_effect = lambda p: he_slide if p == he_path else ihc_slide

        registrar = MockRegistrar()
        mock_HEIHCRegistrar.return_value = registrar

        mock_sample_grid_patches.return_value = [
            (0, 0, 512, 512),
            (512, 512, 512, 512),
        ]

        # DataFrame must include all (patch_id, marker) combinations
        df = pd.DataFrame({
            "patch_id": ["test_case_0000", "test_case_0001"],
            "slide_id": ["test_case", "test_case"],
            "marker": ["CD3", "CD3"],
            "he_x": [0, 512],
            "he_y": [0, 512],
            "he_w": [512, 512],
            "he_h": [512, 512],
            "ihc_x": [10, 522],
            "ihc_y": [20, 532],
            "ihc_w": [512, 512],
            "ihc_h": [512, 512],
            "clipped": [False, False],
        })
        mock_build_mapping_table.return_value = df

        mock_read_patch_rgb.return_value = np.zeros((512, 512, 3), dtype=np.uint8)

        config = {
            "patch_size": 512,
            "stride": 512,
            "he_level": 0,
            "registration_level": 3,
            "max_white_ratio": 1.0,
            "max_image_dim_px": 1024,
            "max_non_rigid_dim_px": 2048,
            "mapping_csv_name": "mapping.csv",
            "viz_sample_n": 2,
        }
        output_dir = tmp_path / "outputs"

        csv_path = run_case(case_dir, config, output_dir)

        assert csv_path.exists()
        result_df = pd.read_csv(csv_path)
        assert len(result_df) == 2
        assert list(result_df.columns) == [
            "patch_id", "slide_id", "marker", "he_x", "he_y", "he_w", "he_h",
            "ihc_x", "ihc_y", "ihc_w", "ihc_h", "clipped",
        ]

        # Verify orchestration
        mock_discover_case.assert_called_once_with(case_dir)
        mock_open_slide.assert_called()
        mock_HEIHCRegistrar.assert_called_once()
        mock_sample_grid_patches.assert_called_once()
        mock_build_mapping_table.assert_called_once()
        mock_create_html_gallery.assert_called_once()

    @patch("he2ihc_align.cli.open_slide")
    @patch("he2ihc_align.cli.HEIHCRegistrar")
    @patch("he2ihc_align.cli.sample_grid_patches")
    @patch("he2ihc_align.cli.build_mapping_table")
    @patch("he2ihc_align.cli.create_html_gallery")
    @patch("he2ihc_align.cli.read_patch_rgb")
    @patch("he2ihc_align.cli.discover_case")
    def test_run_case_no_gallery_when_viz_zero(
        self,
        mock_discover_case,
        mock_read_patch_rgb,
        mock_create_html_gallery,
        mock_build_mapping_table,
        mock_sample_grid_patches,
        mock_HEIHCRegistrar,
        mock_open_slide,
        tmp_path,
    ):
        """Test that gallery is not generated when viz_sample_n is 0."""
        # Setup mocks - create directory structure: parent/case_dir/...
        parent_dir = tmp_path / "test_case"
        parent_dir.mkdir()
        case_dir = parent_dir / "batch"
        case_dir.mkdir()
        he_path = case_dir / "test.kfb"
        he_path.write_text("mock")
        markers = {"CD3": case_dir / "test CD3.svs"}
        for p in markers.values():
            p.write_text("mock")
        mock_discover_case.return_value = (he_path, markers)

        he_slide = MagicMock()
        ihc_slide = MagicMock()
        mock_open_slide.side_effect = lambda p: he_slide if p == he_path else ihc_slide

        registrar = MockRegistrar()
        mock_HEIHCRegistrar.return_value = registrar

        mock_sample_grid_patches.return_value = [
            (0, 0, 512, 512),
        ]

        df = pd.DataFrame({
            "patch_id": ["test_case_0000"],
            "slide_id": ["test_case",],
            "marker": ["CD3"],
            "he_x": [0],
            "he_y": [0],
            "he_w": [512],
            "he_h": [512],
            "ihc_x": [10],
            "ihc_y": [20],
            "ihc_w": [512],
            "ihc_h": [512],
            "clipped": [False],
        })
        mock_build_mapping_table.return_value = df

        config = {
            "patch_size": 512,
            "stride": 512,
            "he_level": 0,
            "registration_level": 3,
            "max_white_ratio": 1.0,
            "max_image_dim_px": 1024,
            "max_non_rigid_dim_px": 2048,
            "mapping_csv_name": "mapping.csv",
            "viz_sample_n": 0,
        }
        output_dir = tmp_path / "outputs"

        run_case(case_dir, config, output_dir)

        gallery_path = output_dir / "gallery.html"
        assert not gallery_path.exists()
        mock_create_html_gallery.assert_not_called()

    @patch("he2ihc_align.cli.open_slide")
    @patch("he2ihc_align.cli.HEIHCRegistrar")
    @patch("he2ihc_align.cli.sample_grid_patches")
    @patch("he2ihc_align.cli.build_mapping_table")
    @patch("he2ihc_align.cli.create_html_gallery")
    @patch("he2ihc_align.cli.read_patch_rgb")
    @patch("he2ihc_align.cli.discover_case")
    def test_run_case_generates_gallery_html(
        self,
        mock_discover_case,
        mock_read_patch_rgb,
        mock_create_html_gallery,
        mock_build_mapping_table,
        mock_sample_grid_patches,
        mock_HEIHCRegistrar,
        mock_open_slide,
        tmp_path,
    ):
        """Test that run_case generates gallery.html when viz_sample_n > 0."""
        # Setup mocks - create directory structure: parent/case_dir/...
        parent_dir = tmp_path / "test_case"
        parent_dir.mkdir()
        case_dir = parent_dir / "batch"
        case_dir.mkdir()
        he_path = case_dir / "test.kfb"
        he_path.write_text("mock")
        markers = {"CD3": case_dir / "test CD3.svs"}
        for p in markers.values():
            p.write_text("mock")
        mock_discover_case.return_value = (he_path, markers)

        he_slide = MagicMock()
        ihc_slide = MagicMock()
        mock_open_slide.side_effect = lambda p: he_slide if p == he_path else ihc_slide

        registrar = MockRegistrar()
        mock_HEIHCRegistrar.return_value = registrar

        mock_sample_grid_patches.return_value = [(0, 0, 512, 512)]

        df = pd.DataFrame({
            "patch_id": ["test_case_0000"],
            "slide_id": ["test_case"],
            "marker": ["CD3"],
            "he_x": [0],
            "he_y": [0],
            "he_w": [512],
            "he_h": [512],
            "ihc_x": [10],
            "ihc_y": [20],
            "ihc_w": [512],
            "ihc_h": [512],
            "clipped": [False],
        })
        mock_build_mapping_table.return_value = df

        mock_read_patch_rgb.return_value = np.zeros((512, 512, 3), dtype=np.uint8)

        config = {
            "patch_size": 512,
            "stride": 512,
            "he_level": 0,
            "registration_level": 3,
            "max_white_ratio": 1.0,
            "max_image_dim_px": 1024,
            "max_non_rigid_dim_px": 2048,
            "mapping_csv_name": "mapping.csv",
            "viz_sample_n": 2,
        }
        output_dir = tmp_path / "outputs"

        run_case(case_dir, config, output_dir)

        mock_create_html_gallery.assert_called_once()
        call_args = mock_create_html_gallery.call_args
        assert call_args[0][0] == output_dir / "gallery.html"


class TestRunCaseReal:
    """Slow integration tests with real slide files."""

    @pytest.mark.slow
    def test_run_case_produces_non_empty_csv(self, tmp_path):
        """Test that run_case produces a non-empty CSV file."""
        case_dir = TEST_DATA / "174162-1" / "174162-1-第一批"
        config = {
            "patch_size": 512,
            "stride": 512,
            "he_level": 3,
            "registration_level": 3,
            "max_white_ratio": 1.0,
            "max_image_dim_px": 1024,
            "max_non_rigid_dim_px": 2048,
            "mapping_csv_name": "mapping.csv",
            "viz_sample_n": 2,
        }
        output_dir = tmp_path / "outputs"

        csv_path = run_case(case_dir, config, output_dir)

        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) > 0
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

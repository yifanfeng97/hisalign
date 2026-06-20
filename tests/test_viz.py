"""Tests for visualization utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from hisalign import viz


class TestSamplePatchIndices:
    """Tests for sample_patch_indices."""

    def test_samples_without_replacement(self):
        indices = viz.sample_patch_indices(10, 3, random_seed=42)
        assert len(indices) == 3
        assert len(set(indices)) == 3
        assert all(0 <= i < 10 for i in indices)

    def test_returns_empty_when_n_patches_zero(self):
        assert viz.sample_patch_indices(0, 5, random_seed=42) == []

    def test_does_not_exceed_available_patches(self):
        indices = viz.sample_patch_indices(2, 5, random_seed=42)
        assert len(indices) == 2

    def test_reproducible_with_seed(self):
        assert viz.sample_patch_indices(
            20, 5, random_seed=123
        ) == viz.sample_patch_indices(20, 5, random_seed=123)

    def test_excludes_clipped_patches_when_requested(self):
        clipped_flags = [True, False, True, False, True]
        indices = viz.sample_patch_indices(
            5, 10, random_seed=42, clipped_flags=clipped_flags, include_clipped=False
        )
        assert all(not clipped_flags[i] for i in indices)
        assert set(indices).issubset({1, 3})


class TestMakePatchFigure:
    """Tests for make_patch_figure."""

    def test_creates_figure(self):
        he_patch = np.zeros((64, 64, 3), dtype=np.uint8)
        ihc_patches = {"CD3": np.zeros((64, 64, 3), dtype=np.uint8)}
        fig = viz.make_patch_figure(he_patch, ihc_patches, "test")
        assert fig is not None

    def test_clipped_flag_adds_badge(self):
        he_patch = np.zeros((64, 64, 3), dtype=np.uint8)
        ihc_patches = {"CD3": np.zeros((64, 64, 3), dtype=np.uint8)}
        fig = viz.make_patch_figure(
            he_patch, ihc_patches, "test", clipped_flags={"CD3": True}
        )
        assert fig is not None


class TestOverlayFigures:
    """Tests for slide-level overlay figures."""

    def test_make_overlay_figure(self):
        he_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        ihc_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        fig = viz.make_overlay_figure(he_img, ihc_img, "overlay")
        assert fig is not None

    def test_make_rigid_overlay_figure(self):
        he_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        ihc_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        m = np.eye(3)
        fig = viz.make_rigid_overlay_figure(he_img, ihc_img, m, "rigid")
        assert fig is not None

    def test_make_non_rigid_overlay_figure(self):
        he_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        ihc_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        nr_reg = MagicMock()
        nr_reg.warp_image.return_value = np.zeros((64, 64), dtype=np.uint8)
        fig = viz.make_non_rigid_overlay_figure(he_img, ihc_img, nr_reg, "non-rigid")
        assert fig is not None


class TestDeformationFieldFigure:
    """Tests for deformation field visualization."""

    def test_creates_figure(self):
        dx = np.zeros((32, 32))
        dy = np.zeros((32, 32))
        fig = viz.make_deformation_field_figure([dx, dy], "field")
        assert fig is not None


class TestMarkerThumbnailFigure:
    """Tests for marker thumbnail figure."""

    def test_creates_figure(self):
        he_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        ihc_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        nr_reg = MagicMock()
        nr_reg.warp_image.return_value = np.zeros((64, 64), dtype=np.uint8)
        fig = viz.make_marker_thumbnail_figure(he_img, ihc_img, nr_reg, "CD3")
        assert fig is not None


class MockRigidRegistrar:
    """Minimal mock for compute_marker_metrics tests."""

    def __init__(self, ref_kp, moving_kp, m=None, n_matches=None):
        self.ref_img = np.zeros((100, 100), dtype=np.uint8)
        self.moving_img = np.zeros((100, 100), dtype=np.uint8)
        self.matched_kp_ref = (
            np.asarray(ref_kp, dtype=np.float64) if ref_kp is not None else None
        )
        self.matched_kp_moving = (
            np.asarray(moving_kp, dtype=np.float64) if moving_kp is not None else None
        )
        self.M = m if m is not None else np.eye(3)
        self.n_matches = (
            n_matches
            if n_matches is not None
            else (len(ref_kp) if ref_kp is not None else 0)
        )

    def inverse_warp_xy(self, xy):
        return np.asarray(xy, dtype=np.float64)

    def warp_xy(self, xy):
        # Translate moving points by -1 in x to align with ref
        xy = np.asarray(xy, dtype=np.float64)
        return xy - np.array([1.0, 0.0])


class MockNonRigidRegistrar:
    """Minimal mock for compute_marker_metrics tests."""

    def __init__(self, warp_offset: float = 1.0):
        self.ref_img = np.zeros((100, 100), dtype=np.uint8)
        self.moving_img = np.zeros((100, 100), dtype=np.uint8)
        self.bk_dxdy = [np.zeros((100, 100)), np.zeros((100, 100))]
        self.warp_offset = warp_offset

    def inverse_warp_xy(self, xy):
        return np.asarray(xy, dtype=np.float64)

    def warp_xy(self, xy):
        # Translate moving points by -warp_offset in x to align with ref
        xy = np.asarray(xy, dtype=np.float64)
        return xy - np.array([self.warp_offset, 0.0])


class TestComputeMarkerMetrics:
    """Tests for compute_marker_metrics."""

    def test_zero_when_no_matches(self):
        rigid_reg = MockRigidRegistrar(None, None)
        nr_reg = MockNonRigidRegistrar()
        metrics = viz.compute_marker_metrics(rigid_reg, nr_reg, mpp=0.5)
        assert metrics["original_displacement_um"] == 0.0
        assert metrics["rigid_displacement_um"] == 0.0
        assert metrics["non_rigid_displacement_um"] == 0.0
        assert metrics["n_matches"] == 0

    def test_computes_distances_in_um(self):
        ref_kp = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float64)
        moving_kp = np.array([[1, 0], [11, 0], [1, 10]], dtype=np.float64)
        rigid_reg = MockRigidRegistrar(ref_kp, moving_kp)
        nr_reg = MockNonRigidRegistrar()
        mpp = 0.25
        metrics = viz.compute_marker_metrics(rigid_reg, nr_reg, mpp=mpp)
        assert metrics["original_displacement_um"] == pytest.approx(1.0 * mpp)
        assert metrics["rigid_displacement_um"] == pytest.approx(0.0)
        assert metrics["non_rigid_displacement_um"] == pytest.approx(0.0)
        assert metrics["n_matches"] == 3

    def test_rtre_uses_common_canvas_diagonal(self):
        ref_kp = np.array([[0, 0], [10, 0]], dtype=np.float64)
        moving_kp = np.array([[1, 0], [11, 0]], dtype=np.float64)
        rigid_reg = MockRigidRegistrar(ref_kp, moving_kp)
        # No additional non-rigid correction, so final error equals original error
        nr_reg = MockNonRigidRegistrar(warp_offset=0.0)
        metrics = viz.compute_marker_metrics(rigid_reg, nr_reg, mpp=1.0)
        diagonal = np.sqrt(100**2 + 100**2)
        expected_rtre = 1.0 / diagonal * 100.0
        assert metrics["rtre"] == pytest.approx(expected_rtre)


class TestComputeOverallMetrics:
    """Tests for compute_overall_metrics."""

    def test_averages_across_markers(self):
        marker_metrics = {
            "CD3": {
                "original_displacement_um": 10.0,
                "rigid_displacement_um": 5.0,
                "non_rigid_displacement_um": 2.0,
                "rtre": 2.0,
                "n_matches": 10,
            },
            "CD20": {
                "original_displacement_um": 20.0,
                "rigid_displacement_um": 10.0,
                "non_rigid_displacement_um": 4.0,
                "rtre": 4.0,
                "n_matches": 20,
            },
        }
        overall = viz.compute_overall_metrics(marker_metrics)
        assert overall["original_displacement_um"] == 15.0
        assert overall["rigid_displacement_um"] == 7.5
        assert overall["non_rigid_displacement_um"] == 3.0
        assert overall["rtre"] == 3.0
        assert overall["n_matches"] == 15


class TestCreateHtmlReport:
    """Tests for create_html_report."""

    def test_writes_html(self, tmp_path):
        output_path = tmp_path / "report.html"
        overall_metrics = {
            "original_displacement_um": 100.0,
            "rigid_displacement_um": 50.0,
            "non_rigid_displacement_um": 25.0,
            "rtre": 3.5,
            "n_matches": 100,
        }
        overlay_entries = [{"title": "未配准", "data_uri": "data:image/png;base64,abc"}]
        marker_rows = [
            {
                "marker": "CD3",
                "original_displacement_um": 100.0,
                "rigid_displacement_um": 50.0,
                "non_rigid_displacement_um": 25.0,
                "rtre": 3.5,
                "n_matches": 100,
                "thumb_uri": "data:image/png;base64,thumb",
                "def_uri": "data:image/png;base64,def",
            }
        ]
        result = viz.create_html_report(
            output_path, "test_case", overall_metrics, overlay_entries, marker_rows
        )
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "配准报告" in content
        assert "CD3" in content
        assert "整体误差统计" in content
        assert "全片叠加对比" in content
        assert "对齐到" in content
        assert "HE" in content

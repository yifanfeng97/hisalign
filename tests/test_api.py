"""Tests for the HisAlign public API and HisAlignModel serialization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hisalign.api import HisAlign, HisAlignModel, warp_xy


class MockModel:
    """Deterministic mock model for offline warp tests."""

    def __init__(self, offset_x: float = 10.0, offset_y: float = 20.0):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.rigid_M = {"CD3": np.eye(3)}
        self.he_padding_matrix = np.eye(3)
        self.ihc_padding_matrix = {"CD3": np.eye(3)}
        self.he_scale_to_level0 = 1.0
        self.ihc_scale_to_level0 = {"CD3": 1.0}
        self.bk_dxdy = {"CD3": None}
        self.fwd_dxdy = {"CD3": None}
        self.he_path = "/tmp/HE.kfb"
        self.ihc_paths = {"CD3": "/tmp/CD3.svs"}

    def warp_xy(self, coords, marker, direction):
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        if direction == "he_to_ihc":
            return coords + np.array([self.offset_x, self.offset_y])
        return coords - np.array([self.offset_x, self.offset_y])


class TestHisAlignModel:
    """Tests for model serialization and offline coordinate mapping."""

    def test_save_and_load_roundtrip(self, tmp_path):
        model = HisAlignModel(
            he_path="/tmp/he.kfb",
            ihc_paths={"CD3": "/tmp/cd3.svs"},
            config={"registration_level": 3},
            he_scale_to_level0=1.0,
            ihc_scale_to_level0={"CD3": 1.0},
            he_padding_matrix=np.eye(3),
            ihc_padding_matrix={"CD3": np.eye(3)},
            rigid_matrix={"CD3": np.eye(3)},
            bk_dxdy={"CD3": None},
            fwd_dxdy={"CD3": None},
            mpp=0.25,
            reg_shape_rc={"CD3": (1024, 1024)},
        )
        path = tmp_path / "model.pkl"
        model.save(path)
        loaded = HisAlignModel.load(path)
        assert loaded.version == model.version
        assert loaded.he_path == model.he_path
        np.testing.assert_array_equal(loaded.he_padding_matrix, model.he_padding_matrix)

    def test_warp_xy_he_to_ihc(self):
        model = MockModel(offset_x=5.0, offset_y=10.0)
        coords = np.array([[0, 0], [100, 100]], dtype=np.float64)
        result = warp_xy(coords, "CD3", "he_to_ihc", model)
        expected = coords + np.array([5.0, 10.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_warp_xy_ihc_to_he(self):
        model = MockModel(offset_x=5.0, offset_y=10.0)
        coords = np.array([[5, 10], [105, 110]], dtype=np.float64)
        result = warp_xy(coords, "CD3", "ihc_to_he", model)
        expected = coords - np.array([5.0, 10.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_warp_xy_unknown_marker_raises(self):
        model = HisAlignModel(
            rigid_matrix={"CD3": np.eye(3)},
            ihc_scale_to_level0={"CD3": 1.0},
            ihc_padding_matrix={"CD3": np.eye(3)},
        )
        with pytest.raises(KeyError, match="Marker 'Ki67' not found"):
            model.warp_xy([[0, 0]], marker="Ki67", direction="he_to_ihc")

    def test_warp_xy_invalid_direction_raises(self):
        model = HisAlignModel(
            rigid_matrix={"CD3": np.eye(3)},
            ihc_scale_to_level0={"CD3": 1.0},
            ihc_padding_matrix={"CD3": np.eye(3)},
        )
        with pytest.raises(ValueError, match="direction must be"):
            model.warp_xy([[0, 0]], marker="CD3", direction="invalid")


class TestHisAlign:
    """Tests for the high-level HisAlign API."""

    @patch("hisalign.api.open_slide")
    @patch("hisalign.api.HEIHCRegistrar")
    def test_fit_returns_model(self, mock_registrar_cls, mock_open_slide):
        he_slide = MagicMock()
        he_slide.level_downsamples = [1.0, 2.0, 4.0, 8.0]
        ihc_slide = MagicMock()
        ihc_slide.level_downsamples = [1.0, 2.0, 4.0, 8.0]
        mock_open_slide.side_effect = [he_slide, ihc_slide]

        mock_registrar = MagicMock()
        mock_registrar.he_scale_to_level0 = 1.0
        mock_registrar.ihc_scale_to_level0 = {"CD3": 1.0}
        mock_registrar.he_padding_matrix = np.eye(3)
        mock_registrar.ihc_padding_matrix = {"CD3": np.eye(3)}
        mock_registrar.reg_shape_rc = {"CD3": (1024, 1024)}
        mock_registrar.rigid_registrars = {"CD3": MagicMock(M=np.eye(3))}
        mock_registrar.non_rigid_registrars = {
            "CD3": MagicMock(bk_dxdy=None, fwd_dxdy=None)
        }
        mock_registrar.fit.return_value = mock_registrar
        mock_registrar_cls.return_value = mock_registrar

        aligner = HisAlign(
            he_path="HE.kfb",
            ihc_paths={"CD3": "CD3.svs"},
            registration_level=3,
        )
        model = aligner.fit()

        assert isinstance(model, HisAlignModel)
        assert model.he_path == str(Path("HE.kfb").resolve())
        assert "CD3" in model.ihc_paths
        he_slide.close.assert_called_once()
        ihc_slide.close.assert_called_once()

    def test_normalize_ihc_paths_from_list(self):
        aligner = HisAlign(he_path="HE.kfb", ihc_paths=["markerA.svs", "markerB.svs"])
        assert set(aligner.ihc_paths.keys()) == {"markerA", "markerB"}

    def test_normalize_ihc_paths_from_dict(self):
        aligner = HisAlign(he_path="HE.kfb", ihc_paths={"CD3": "CD3.svs"})
        assert aligner.ihc_paths["CD3"] == Path("CD3.svs").resolve()

    def test_warp_before_fit_raises(self):
        aligner = HisAlign(he_path="HE.kfb", ihc_paths={"CD3": "CD3.svs"})
        with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
            aligner.warp_xy_from_he_to_ihc([[0, 0]], marker="CD3")


class TestHisAlignReal:
    """Slow integration tests with real slide files."""

    TEST_DATA = Path("/home/fengyifan/disk/code/valis/test_SCCE")

    @pytest.mark.slow
    def test_fit_on_real_slides(self, tmp_path):
        he_path = self.TEST_DATA / "174162-1" / "174162-1-第一批" / "174162-1.kfb"
        ihc_path = self.TEST_DATA / "174162-1" / "174162-1-第一批" / "174162-1 CD3.svs"
        if not he_path.exists() or not ihc_path.exists():
            pytest.skip("Real slide files not available")

        aligner = HisAlign(
            he_path=he_path,
            ihc_paths={"CD3": ihc_path},
            registration_level=3,
            max_image_dim_px=1024,
            preprocessing="od",
            feature_detector="kaze",
            feature_n_levels=3,
            match_max_ratio=1.0,
            mpp=0.25,
        )
        model = aligner.fit()
        assert isinstance(model, HisAlignModel)
        assert model.he_path == str(he_path.resolve())
        assert "CD3" in model.ihc_paths

        out_path = tmp_path / "model.pkl"
        model.save(out_path)
        assert out_path.exists()

        loaded = HisAlignModel.load(out_path)
        assert "CD3" in loaded.rigid_matrix

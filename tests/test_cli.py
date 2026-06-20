"""Tests for the hisalign CLI subcommands."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hisalign.cli import _cmd_register, _cmd_visualize, _cmd_warp, _parse_ihc_args


class MockModel:
    """Deterministic mock model for CLI warp tests."""

    def __init__(self, offset_x: float = 10.0, offset_y: float = 20.0):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.config = {"registration_level": 3, "feature_detector": "kaze"}
        self.he_path = "/tmp/HE.kfb"
        self.ihc_paths = {"CD3": "/tmp/CD3.svs"}

    def save(self, path: Path) -> None:
        Path(path).write_bytes(b"mock")

    def warp_xy(self, coords, marker, direction):
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        if direction == "he_to_ihc":
            return coords + np.array([self.offset_x, self.offset_y])
        return coords - np.array([self.offset_x, self.offset_y])


def test_parse_ihc_args_with_marker():
    args = ["CD3=/path/to/CD3.svs", "Ki67=/path/to/Ki67.svs"]
    result = _parse_ihc_args(args)
    assert set(result.keys()) == {"CD3", "Ki67"}
    assert result["CD3"].name == "CD3.svs"


def test_parse_ihc_args_without_marker(tmp_path):
    cd3 = tmp_path / "slide CD3.svs"
    cd3.write_text("mock")
    ki67 = tmp_path / "slide Ki67.svs"
    ki67.write_text("mock")
    result = _parse_ihc_args([str(cd3), str(ki67)])
    assert set(result.keys()) == {"CD3", "Ki67"}


def test_parse_ihc_args_duplicate_marker_raises():
    with pytest.raises(ValueError, match="Duplicate marker name 'CD3'"):
        _parse_ihc_args(["CD3=a.svs", "CD3=b.svs"])


class TestRegisterCommand:
    """Fast mocked tests for `hisalign register`."""

    @patch("hisalign.cli._generate_visualizations")
    @patch("hisalign.cli.HisAlign")
    def test_register_saves_model(self, mock_hisalign_cls, mock_generate_viz, tmp_path):
        model = MockModel()
        aligner = MagicMock()
        aligner.fit.return_value = model
        aligner._registrar = MagicMock()
        mock_hisalign_cls.return_value = aligner

        he_path = tmp_path / "HE.kfb"
        he_path.write_text("mock")
        ihc_path = tmp_path / "CD3.svs"
        ihc_path.write_text("mock")
        output_path = tmp_path / "model.pkl"

        config_path = tmp_path / "config.yaml"
        config_path.write_text("generate_report: false\nviz_sample_n: 0\n")

        args = SimpleNamespace(
            he=he_path,
            ihc=[f"CD3={ihc_path}"],
            output=output_path,
            config=config_path,
            mpp=None,
        )

        _cmd_register(args)

        mock_hisalign_cls.assert_called_once()
        assert output_path.exists()
        mock_generate_viz.assert_not_called()

    @patch("hisalign.cli._generate_visualizations")
    @patch("hisalign.cli.HisAlign")
    def test_register_triggers_visualizations(
        self, mock_hisalign_cls, mock_generate_viz, tmp_path
    ):
        model = MockModel()
        aligner = MagicMock()
        aligner.fit.return_value = model
        aligner._registrar = MagicMock()
        mock_hisalign_cls.return_value = aligner

        he_path = tmp_path / "HE.kfb"
        he_path.write_text("mock")
        ihc_path = tmp_path / "CD3.svs"
        ihc_path.write_text("mock")
        output_path = tmp_path / "model.pkl"

        config_path = tmp_path / "config.yaml"
        config_path.write_text("viz_sample_n: 2\ngenerate_report: true\n")

        args = SimpleNamespace(
            he=he_path,
            ihc=[f"CD3={ihc_path}"],
            output=output_path,
            config=config_path,
            mpp=None,
        )

        _cmd_register(args)

        mock_generate_viz.assert_called_once()


class TestWarpCommand:
    """Fast mocked tests for `hisalign warp`."""

    @patch("hisalign.cli.HisAlignModel.load")
    def test_warp_writes_mapped_csv(self, mock_load, tmp_path):
        model = MockModel(offset_x=5.0, offset_y=10.0)
        mock_load.return_value = model

        coords_path = tmp_path / "coords.csv"
        pd.DataFrame({"x": [0, 100], "y": [0, 100]}).to_csv(coords_path, index=False)
        output_path = tmp_path / "mapped.csv"

        args = SimpleNamespace(
            model=tmp_path / "model.pkl",
            marker="CD3",
            direction="he_to_ihc",
            coords=coords_path,
            output=output_path,
        )

        _cmd_warp(args)

        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert list(df.columns) == ["x", "y", "marker", "direction"]
        assert df["x"].tolist() == [5.0, 105.0]
        assert df["y"].tolist() == [10.0, 110.0]


class TestVisualizeCommand:
    """Fast mocked tests for `hisalign visualize`."""

    @patch("hisalign.cli._generate_visualizations")
    @patch("hisalign.cli.HisAlign")
    @patch("hisalign.cli.HisAlignModel.load")
    def test_visualize_reloads_model_and_generates_outputs(
        self, mock_load, mock_hisalign_cls, mock_generate_viz, tmp_path
    ):
        model = MockModel()
        mock_load.return_value = model

        aligner = MagicMock()
        aligner._registrar = MagicMock()
        mock_hisalign_cls.return_value = aligner

        model_path = tmp_path / "model.pkl"
        output_dir = tmp_path / "out"

        args = SimpleNamespace(
            model=model_path,
            config=tmp_path / "nonexistent.yaml",
            output_dir=output_dir,
        )

        _cmd_visualize(args)

        mock_load.assert_called_once_with(model_path)
        mock_hisalign_cls.assert_called_once()
        mock_generate_viz.assert_called_once()

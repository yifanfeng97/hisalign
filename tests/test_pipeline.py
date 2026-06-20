"""End-to-end integration tests using the new HisAlign public API."""

from __future__ import annotations

from pathlib import Path

import pytest

from hisalign import HisAlign, HisAlignModel

TEST_DATA = Path("/home/fengyifan/disk/code/valis/test_SCCE")


@pytest.mark.slow
class TestPipelineEndToEnd:
    """Slow integration tests that run the full pipeline on real slide data."""

    def test_hisalign_fit_and_save_model(self, tmp_path):
        """Run HisAlign.fit() on real slides and verify the saved model reloads."""
        case_dir = TEST_DATA / "174162-1" / "174162-1-第一批"
        if not case_dir.exists():
            pytest.skip(f"Real test data not found at {case_dir}")

        he_path = case_dir / "174162-1.kfb"
        ihc_paths = sorted(case_dir.glob("*.svs"))
        if not he_path.exists() or not ihc_paths:
            pytest.skip("HE or IHC files not found")

        ihc_dict = {p.stem.split()[-1]: p for p in ihc_paths}

        aligner = HisAlign(
            he_path=he_path,
            ihc_paths=ihc_dict,
            registration_level=3,
            max_image_dim_px=512,
            preprocessing="od",
            feature_detector="kaze",
            feature_n_levels=3,
            match_max_ratio=1.0,
            mpp=0.25,
        )
        model = aligner.fit()
        assert isinstance(model, HisAlignModel)
        assert model.he_path == str(he_path.resolve())
        assert set(model.ihc_paths.keys()) == set(ihc_dict.keys())

        model_path = tmp_path / "model.pkl"
        model.save(model_path)
        loaded = HisAlignModel.load(model_path)
        assert loaded.version == model.version

        # Sanity-check offline coordinate mapping
        coords = [[1000.0, 2000.0]]
        for marker in ihc_dict:
            mapped = loaded.warp_xy(coords, marker=marker, direction="he_to_ihc")
            assert mapped.shape == (1, 2)

"""End-to-end integration test for the full HE-to-IHC alignment pipeline.

Runs the pipeline on real data with small parameters to keep runtime reasonable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from he2ihc_align.cli import run_case

TEST_DATA = Path("/home/fengyifan/disk/code/valis/test_SCCE")


@pytest.mark.slow
class TestPipelineEndToEnd:
    """Slow integration tests that run the full pipeline on real slide data."""

    def test_run_case_produces_non_empty_csv(self, tmp_path):
        """Run the full pipeline on a real case and verify output CSV is non-empty."""
        case_dir = TEST_DATA / "174162-1" / "174162-1-第一批"
        if not case_dir.exists():
            pytest.skip(f"Real test data not found at {case_dir}")

        config = {
            "patch_size": 256,
            "stride": 256,
            "he_level": 3,
            "registration_level": 3,
            "max_white_ratio": 1.0,
            "max_image_dim_px": 512,
            "max_non_rigid_dim_px": 512,
            "mapping_csv_name": "mapping.csv",
            "viz_sample_n": 1,
        }
        output_dir = tmp_path / "outputs"

        csv_path = run_case(case_dir, config, output_dir)

        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) > 0, "Expected non-empty mapping CSV"

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

        # Verify that we have at least one row per marker
        markers = df["marker"].unique()
        assert len(markers) > 0

        # Verify patch_id format
        assert df["patch_id"].iloc[0].startswith("174162-1_")

    def test_run_case_output_dir_created(self, tmp_path):
        """Verify that the output directory is created if it does not exist."""
        case_dir = TEST_DATA / "174162-1" / "174162-1-第一批"
        if not case_dir.exists():
            pytest.skip(f"Real test data not found at {case_dir}")

        config = {
            "patch_size": 256,
            "stride": 256,
            "he_level": 3,
            "registration_level": 3,
            "max_white_ratio": 1.0,
            "max_image_dim_px": 512,
            "max_non_rigid_dim_px": 512,
            "mapping_csv_name": "mapping.csv",
            "viz_sample_n": 0,
        }
        output_dir = tmp_path / "new_outputs" / "nested"

        csv_path = run_case(case_dir, config, output_dir)

        assert output_dir.exists()
        assert csv_path.exists()

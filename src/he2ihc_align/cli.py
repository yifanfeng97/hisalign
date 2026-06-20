"""Command-line interface for HE-to-IHC alignment pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from he2ihc_align.case_io import discover_case
from he2ihc_align.mapping import build_mapping_table
from he2ihc_align.patching import sample_grid_patches
from he2ihc_align.registration.registrar import HEIHCRegistrar
from he2ihc_align.slide_io.factory import open_slide
from he2ihc_align.viz import (
    create_html_gallery,
    fig_to_data_uri,
    make_patch_figure,
    read_patch_rgb,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_case(case_dir: Path, config: dict, output_dir: Path) -> Path:
    """Run full pipeline for one case batch directory.

    Discovers HE/IHC slides, runs registration, samples patches, builds
    mapping table, writes CSV, and optionally generates an HTML gallery.

    Parameters
    ----------
    case_dir : Path
        Path to the case batch directory (e.g. .../174162-1/174162-1-第一批).
    config : dict
        Configuration dictionary.
    output_dir : Path
        Directory to write outputs.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover HE and IHC slides directly in the batch directory
    case_dir = Path(case_dir)
    he_path, markers = discover_case(case_dir)
    # Derive case_id from HE path: if HE is in a batch subdir, use parent.parent;
    # otherwise use parent (case directory itself)
    batch_dir_name = config.get("batch_dir_name", "第一批")
    if he_path.parent.name.endswith(batch_dir_name):
        case_id = he_path.parent.parent.name
    else:
        case_id = he_path.parent.name

    logger.info("Case %s: HE=%s, markers=%s", case_id, he_path.name, list(markers.keys()))

    # Open slides
    he_slide = open_slide(he_path)
    ihc_slides = {marker: open_slide(path) for marker, path in markers.items()}

    try:
        # Registration
        registrar = HEIHCRegistrar(
            he_slide=he_slide,
            ihc_slides=ihc_slides,
            registration_level=config.get("registration_level", 3),
            max_image_dim_px=config.get("max_image_dim_px", 1024),
            max_non_rigid_dim_px=config.get("max_non_rigid_dim_px", 2048),
        )
        registrar.fit()
        logger.info("Registration complete for %s", case_id)

        # Sample patches
        he_patch_bboxes = sample_grid_patches(
            he_slide,
            patch_size=config.get("patch_size", 512),
            stride=config.get("stride", 512),
            level=config.get("he_level", 0),
            max_white_ratio=config.get("max_white_ratio", 0.95),
        )
        logger.info("Sampled %d patches for %s", len(he_patch_bboxes), case_id)

        # Build mapping table
        # bboxes from sample_grid_patches are always in level-0 coordinates
        df = build_mapping_table(
            registrar=registrar,
            he_slide=he_slide,
            ihc_slides=ihc_slides,
            he_patch_bboxes=he_patch_bboxes,
            slide_id=case_id,
            he_level=0,
            ihc_level=0,
        )

        # Write CSV
        csv_path = output_dir / config.get("mapping_csv_name", "mapping.csv")
        df.to_csv(csv_path, index=False)
        logger.info("Wrote mapping CSV to %s", csv_path)

        # Generate HTML gallery for viz_sample_n patches
        viz_sample_n = config.get("viz_sample_n", 5)
        if viz_sample_n > 0 and len(he_patch_bboxes) > 0:
            gallery_entries = []
            sample_indices = range(min(viz_sample_n, len(he_patch_bboxes)))

            for idx in sample_indices:
                he_x, he_y, he_w, he_h = he_patch_bboxes[idx]
                he_patch = read_patch_rgb(he_slide, he_x, he_y, he_w, he_h, level=0)

                ihc_patches = {}
                for marker in ihc_slides:
                    rows = df[(df["patch_id"] == f"{case_id}_{idx:04d}") & (df["marker"] == marker)]
                    if rows.empty:
                        continue
                    row = rows.iloc[0]
                    ihc_x, ihc_y, ihc_w, ihc_h = int(row["ihc_x"]), int(row["ihc_y"]), int(row["ihc_w"]), int(row["ihc_h"])
                    if ihc_w > 0 and ihc_h > 0:
                        ihc_patch = read_patch_rgb(ihc_slides[marker], ihc_x, ihc_y, ihc_w, ihc_h, level=0)
                    else:
                        # Degenerate bbox: create a blank placeholder
                        ihc_patch = np.zeros((he_h, he_w, 3), dtype=np.uint8)
                    ihc_patches[marker] = ihc_patch

                title = f"{case_id} patch {idx:04d} (HE: {he_x},{he_y} {he_w}x{he_h})"
                fig = make_patch_figure(he_patch, ihc_patches, title)
                data_uri = fig_to_data_uri(fig)
                gallery_entries.append({"title": title, "data_uri": data_uri})

            gallery_path = output_dir / "gallery.html"
            create_html_gallery(gallery_path, case_id, gallery_entries)
            logger.info("Wrote gallery HTML to %s", gallery_path)

    finally:
        he_slide.close()
        for s in ihc_slides.values():
            s.close()

    return csv_path


def main() -> None:
    """CLI entry point for he2ihc-align."""
    parser = argparse.ArgumentParser(description="HE-to-IHC whole-slide image alignment pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Path to config YAML")
    parser.add_argument("--case-dir", type=Path, default=None, help="Run only this case batch directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory override")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir) if args.output_dir else Path(config.get("output_root", "./outputs"))

    if args.case_dir:
        # Run single case
        run_case(args.case_dir, config, output_dir)
    else:
        # Iterate over data_root and run each case batch directory
        data_root = Path(config.get("data_root", "."))
        batch_dir_name = config.get("batch_dir_name", "第一批")

        for case_dir in sorted(data_root.iterdir()):
            if not case_dir.is_dir():
                continue
            batch_dir = case_dir / batch_dir_name
            if not batch_dir.exists():
                logger.warning("Skipping %s: no batch directory %s", case_dir, batch_dir_name)
                continue
            try:
                run_case(batch_dir, config, output_dir / case_dir.name)
            except Exception as e:
                logger.error("Failed to process %s: %s", case_dir, e)
                continue


if __name__ == "__main__":
    main()

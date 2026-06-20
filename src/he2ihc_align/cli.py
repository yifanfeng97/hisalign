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
from he2ihc_align.slide_io.base import Slide
from he2ihc_align.slide_io.factory import open_slide
from he2ihc_align.viz import (
    compute_marker_metrics,
    compute_overall_metrics,
    create_html_gallery,
    create_html_report,
    fig_to_data_uri,
    make_deformation_field_figure,
    make_marker_thumbnail_figure,
    make_non_rigid_overlay_figure,
    make_overlay_figure,
    make_patch_figure,
    make_rigid_overlay_figure,
    read_patch_rgb,
    read_slide_thumbnail,
    sample_patch_indices,
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

        # Generate HTML gallery for viz_sample_n randomly chosen patches
        viz_sample_n = config.get("viz_sample_n", 5)
        if viz_sample_n > 0 and len(he_patch_bboxes) > 0:
            # Build per-patch clipped flags for each marker
            clipped_per_patch = [
                df[df["patch_id"] == f"{case_id}_{idx:04d}"]["clipped"].tolist()
                for idx in range(len(he_patch_bboxes))
            ]
            patch_is_clipped = [any(flags) for flags in clipped_per_patch]

            sample_indices = sample_patch_indices(
                n_patches=len(he_patch_bboxes),
                viz_sample_n=viz_sample_n,
                random_seed=config.get("viz_random_seed"),
                clipped_flags=patch_is_clipped,
                include_clipped=config.get("viz_sample_clipped", False),
            )

            gallery_entries = []
            for idx in sample_indices:
                he_x, he_y, he_w, he_h = he_patch_bboxes[idx]
                he_patch = read_patch_rgb(he_slide, he_x, he_y, he_w, he_h, level=0)

                ihc_patches = {}
                clipped_flags = {}
                for marker in ihc_slides:
                    rows = df[(df["patch_id"] == f"{case_id}_{idx:04d}") & (df["marker"] == marker)]
                    if rows.empty:
                        continue
                    row = rows.iloc[0]
                    ihc_x, ihc_y = int(row["ihc_x"]), int(row["ihc_y"])
                    ihc_w, ihc_h = int(row["ihc_w"]), int(row["ihc_h"])
                    clipped_flags[marker] = bool(row["clipped"])
                    if ihc_w > 0 and ihc_h > 0:
                        ihc_patch = read_patch_rgb(ihc_slides[marker], ihc_x, ihc_y, ihc_w, ihc_h, level=0)
                    else:
                        # Degenerate bbox: create a blank placeholder
                        ihc_patch = np.zeros((he_h, he_w, 3), dtype=np.uint8)
                    ihc_patches[marker] = ihc_patch

                title = f"{case_id} patch {idx:04d} (HE: {he_x},{he_y} {he_w}x{he_h})"
                fig = make_patch_figure(he_patch, ihc_patches, title, clipped_flags=clipped_flags)
                data_uri = fig_to_data_uri(fig)
                gallery_entries.append({"title": title, "data_uri": data_uri})

            gallery_path = output_dir / "gallery.html"
            create_html_gallery(gallery_path, case_id, gallery_entries)
            logger.info("Wrote gallery HTML to %s", gallery_path)

        # Generate slide-level registration quality report
        if config.get("generate_report", True):
            report_path = output_dir / "report.html"
            _generate_report(
                report_path=report_path,
                slide_id=case_id,
                registrar=registrar,
                he_slide=he_slide,
                ihc_slides=ihc_slides,
                report_level=config.get("report_level", 3),
                max_report_dim_px=config.get("max_report_dim_px", 2048),
            )
            logger.info("Wrote report HTML to %s", report_path)

    finally:
        he_slide.close()
        for s in ihc_slides.values():
            s.close()

    return csv_path


def _generate_report(
    report_path: Path,
    slide_id: str,
    registrar: HEIHCRegistrar,
    he_slide: Slide,
    ihc_slides: dict[str, Slide],
    report_level: int,
    max_report_dim_px: int,
) -> Path:
    """Generate the slide-level registration quality report."""
    # Read thumbnails
    he_img = read_slide_thumbnail(he_slide, report_level, max_dim_px=max_report_dim_px)

    marker_rows = []
    overlay_entries = []
    first_marker = True

    for marker, ihc_slide in ihc_slides.items():
        ihc_img = read_slide_thumbnail(ihc_slide, report_level, max_dim_px=max_report_dim_px)

        rigid_reg = registrar.rigid_registrars[marker]
        nr_reg = registrar.non_rigid_registrars[marker]

        # Overlays for the first marker (representative global view)
        if first_marker:
            fig = make_overlay_figure(he_img, ihc_img, "Unregistered")
            overlay_entries.append({"title": "Unregistered", "data_uri": fig_to_data_uri(fig)})

            fig = make_rigid_overlay_figure(he_img, ihc_img, rigid_reg.M, "Rigid")
            overlay_entries.append({"title": "Rigid", "data_uri": fig_to_data_uri(fig)})

            fig = make_non_rigid_overlay_figure(he_img, ihc_img, nr_reg, "Non-rigid")
            overlay_entries.append({"title": "Non-rigid", "data_uri": fig_to_data_uri(fig)})
            first_marker = False

        # Marker thumbnail
        fig = make_marker_thumbnail_figure(he_img, ihc_img, nr_reg, marker)
        thumb_uri = fig_to_data_uri(fig)

        # Deformation field
        if nr_reg.bk_dxdy is not None:
            fig = make_deformation_field_figure(nr_reg.bk_dxdy, f"{marker} deformation field")
            def_uri = fig_to_data_uri(fig)
        else:
            def_uri = ""

        # Metrics
        metrics = compute_marker_metrics(
            rigid_registrar=rigid_reg,
            non_rigid_registrar=nr_reg,
            he_scale=registrar.he_scale_to_level0,
            nr_he_scale=max(he_slide.level_dimensions[0]) / max(nr_reg.ref_img.shape[:2]),
        )

        marker_rows.append(
            {
                "marker": marker,
                "original_displacement_px": metrics["original_displacement_px"],
                "rigid_displacement_px": metrics["rigid_displacement_px"],
                "non_rigid_displacement_px": metrics["non_rigid_displacement_px"],
                "rtre": metrics["rtre"],
                "n_matches": metrics["n_matches"],
                "thumb_uri": thumb_uri,
                "def_uri": def_uri,
            }
        )

    overall_metrics = compute_overall_metrics({row["marker"]: row for row in marker_rows})

    return create_html_report(
        output_path=report_path,
        slide_id=slide_id,
        overall_metrics=overall_metrics,
        overlay_entries=overlay_entries,
        marker_rows=marker_rows,
    )


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

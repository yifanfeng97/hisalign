"""Command-line interface for hisalign."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from hisalign.api import HisAlign, HisAlignModel
from hisalign.mapping import build_mapping_table
from hisalign.patching import sample_grid_patches
from hisalign.registration import warp_tools
from hisalign.slide_io.factory import get_slide_mpp, open_slide
from hisalign.viz import (
    compute_marker_metrics,
    compute_overall_metrics,
    create_html_report,
    fig_to_data_uri,
    make_deformation_field_figure,
    make_non_rigid_overlay_figure,
    make_overlay_figure,
    make_patch_with_context_figure,
    make_rigid_overlay_figure,
    make_single_thumbnail_figure,
    read_global_thumbnail,
    read_patch_rgb,
    sample_patch_indices,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _downsample_for_web(img: np.ndarray, max_px: int = 384) -> np.ndarray:
    """Resize an image so its longest side is at most ``max_px`` pixels."""
    if max_px <= 0:
        return img
    h, w = img.shape[:2]
    scale = min(1.0, max_px / max(h, w))
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        img = warp_tools.resize_img(img, (new_h, new_w))
    return img


def _load_config(config_path: Path | None) -> dict:
    """Load a YAML config or return an empty dict."""
    if config_path is None or not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_ihc_args(ihc_args: list[str]) -> dict[str, Path]:
    """Parse --ihc values into ``{marker: path}``.

    Accepts either ``marker=path`` or a bare path from which the marker name is
    derived from the filename stem's last whitespace-separated token.
    """
    result: dict[str, Path] = {}
    for arg in ihc_args:
        if "=" in arg:
            marker, path_str = arg.split("=", 1)
            marker = marker.strip()
            path = Path(path_str.strip()).resolve()
        else:
            path = Path(arg).resolve()
            parts = path.stem.split()
            marker = parts[-1] if parts else path.stem
        if marker in result:
            raise ValueError(f"Duplicate marker name '{marker}'")
        result[marker] = path
    return result


def _slide_id_from_he(he_path: Path) -> str:
    """Derive a slide identifier from the HE filename."""
    return he_path.stem


def _generate_visualizations(
    registrar,
    he_path: Path,
    ihc_paths: dict[str, Path],
    config: dict,
    output_dir: Path,
    slide_id: str,
) -> None:
    """Generate optional patch gallery and slide-level report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    he_slide = open_slide(he_path)
    ihc_slides = {marker: open_slide(path) for marker, path in ihc_paths.items()}

    try:
        mpp_level0 = config.get("mpp")
        if mpp_level0 is None:
            mpp_level0 = get_slide_mpp(he_slide)
        if mpp_level0 is None:
            raise ValueError(
                "MPP (microns per pixel) is not available in slide metadata. "
                "Please set 'mpp' in the config YAML or pass --mpp."
            )
        mpp_canvas = mpp_level0 * registrar.he_scale_to_level0

        viz_sample_n = config.get("viz_sample_n", 0)
        gallery_entries: list[dict] = []
        if viz_sample_n and viz_sample_n > 0:
            max_dim_px = config.get("viz_global_thumb_max_dim_px", 512)
            he_global_thumb, he_downsample = read_global_thumbnail(
                he_slide, max_dim_px=max_dim_px
            )
            ihc_global_thumbs = {}
            ihc_downsamples = {}
            for marker, slide in ihc_slides.items():
                thumb, downsample = read_global_thumbnail(slide, max_dim_px=max_dim_px)
                ihc_global_thumbs[marker] = thumb
                ihc_downsamples[marker] = downsample

            he_patch_bboxes = sample_grid_patches(
                he_slide,
                patch_size=config.get("patch_size", 512),
                stride=config.get("stride", 512),
                level=config.get("he_level", 0),
                max_white_ratio=config.get("max_white_ratio", 0.95),
            )
            logger.info("Sampled %d patches for gallery", len(he_patch_bboxes))

            patch_max_px = config.get("viz_patch_max_px", 384)
            patch_fmt = config.get(
                "viz_patch_image_format", config.get("viz_image_format", "png")
            )
            patch_dpi = config.get("viz_patch_dpi", 100)
            patch_quality = config.get(
                "viz_patch_image_quality", config.get("viz_image_quality", 85)
            )
            patch_quality = patch_quality if patch_fmt != "png" else None
            col_inches = config.get("viz_patch_col_inches", 2.0)

            if he_patch_bboxes:
                df = build_mapping_table(
                    registrar=registrar,
                    he_slide=he_slide,
                    ihc_slides=ihc_slides,
                    he_patch_bboxes=he_patch_bboxes,
                    slide_id=slide_id,
                )

                clipped_per_patch = [
                    df[df["patch_id"] == f"{slide_id}_{idx:04d}"]["clipped"].tolist()
                    for idx in range(len(he_patch_bboxes))
                ]
                patch_is_clipped = [any(flags) for flags in clipped_per_patch]

                sample_indices = sample_patch_indices(
                    n_patches=len(he_patch_bboxes),
                    viz_sample_n=viz_sample_n,
                    random_seed=config.get("viz_random_seed"),
                    clipped_flags=patch_is_clipped,
                    include_clipped=config.get("viz_sample_clipped", True),
                )

                for idx in sample_indices:
                    he_x, he_y, he_w, he_h = he_patch_bboxes[idx]
                    he_patch = read_patch_rgb(he_slide, he_x, he_y, he_w, he_h, level=0)
                    he_patch = _downsample_for_web(he_patch, max_px=patch_max_px)

                    ihc_patches: dict[str, np.ndarray] = {}
                    ihc_bboxes_xywh: dict[str, tuple[int, int, int, int]] = {}
                    clipped_flags: dict[str, bool] = {}
                    for marker in ihc_slides:
                        rows = df[
                            (df["patch_id"] == f"{slide_id}_{idx:04d}")
                            & (df["marker"] == marker)
                        ]
                        if rows.empty:
                            continue
                        row = rows.iloc[0]
                        ihc_x, ihc_y = int(row["ihc_x"]), int(row["ihc_y"])
                        ihc_w, ihc_h = int(row["ihc_w"]), int(row["ihc_h"])
                        clipped_flags[marker] = bool(row["clipped"])
                        ihc_bboxes_xywh[marker] = (ihc_x, ihc_y, ihc_w, ihc_h)
                        if ihc_w > 0 and ihc_h > 0:
                            ihc_patch = read_patch_rgb(
                                ihc_slides[marker], ihc_x, ihc_y, ihc_w, ihc_h, level=0
                            )
                            ihc_patch = _downsample_for_web(
                                ihc_patch, max_px=patch_max_px
                            )
                        else:
                            ihc_patch = np.zeros((he_h, he_w, 3), dtype=np.uint8)
                        ihc_patches[marker] = ihc_patch

                    title = (
                        f"{slide_id} patch {idx:04d} (HE: {he_x},{he_y} {he_w}x{he_h})"
                    )
                    fig = make_patch_with_context_figure(
                        he_global_thumb=he_global_thumb,
                        he_downsample=he_downsample,
                        he_bbox_xywh=(he_x, he_y, he_w, he_h),
                        ihc_global_thumbs=ihc_global_thumbs,
                        ihc_downsamples=ihc_downsamples,
                        ihc_bboxes_xywh=ihc_bboxes_xywh,
                        he_patch=he_patch,
                        ihc_patches=ihc_patches,
                        title=title,
                        clipped_flags=clipped_flags,
                        col_inches=col_inches,
                    )
                    data_uri = fig_to_data_uri(
                        fig, fmt=patch_fmt, dpi=patch_dpi, quality=patch_quality
                    )
                    gallery_entries.append({"title": title, "data_uri": data_uri})

        if config.get("generate_report", True):
            report_path = output_dir / "viz_report.html"
            report_data = _generate_report(
                slide_id=slide_id,
                registrar=registrar,
                mpp_canvas=mpp_canvas,
                rtre_threshold=config.get("report_rtre_threshold", 5.0),
                config=config,
            )
            create_html_report(
                output_path=report_path,
                slide_id=slide_id,
                overall_metrics=report_data["overall_metrics"],
                overlay_entries=report_data["overlay_entries"],
                marker_rows=report_data["marker_rows"],
                report_path=report_path,
                rtre_threshold=config.get("report_rtre_threshold", 5.0),
                he_ref_thumb_uri=report_data["he_ref_thumb_uri"],
                gallery_entries=gallery_entries,
            )
            logger.info("Wrote unified visualization report to %s", report_path)
    finally:
        he_slide.close()
        for s in ihc_slides.values():
            s.close()


def _generate_report(
    slide_id: str,
    registrar,
    mpp_canvas: float,
    rtre_threshold: float,
    config: dict,
) -> dict:
    """Build all slide-level report data structures without writing HTML.

    Returns
    -------
    dict
        Keys: ``overall_metrics``, ``overlay_entries``, ``marker_rows``,
        ``he_ref_thumb_uri``.
    """
    he_img = registrar.he_padded
    if he_img is None:
        raise RuntimeError(
            "Registrar has no padded HE image; registration was not run."
        )

    base_fmt = config.get("viz_image_format", "png")
    base_quality = config.get("viz_image_quality", 85)
    overlay_fmt = base_fmt
    overlay_dpi = config.get("viz_overlay_dpi", 100)
    overlay_quality = base_quality if overlay_fmt != "png" else None
    thumb_fmt = base_fmt
    thumb_dpi = config.get("viz_thumb_dpi", 100)
    thumb_quality = base_quality if thumb_fmt != "png" else None
    def_fmt = base_fmt
    def_dpi = config.get("viz_deformation_dpi", 100)
    def_quality = base_quality if def_fmt != "png" else None

    he_ref_shape_rc = registrar.he_padded.shape[0:2]
    he_padding_for_ref = warp_tools.get_padding_matrix(
        registrar.he_gray.shape, he_ref_shape_rc
    )
    he_color_ref = warp_tools.warp_img(
        registrar.he_img,
        M=he_padding_for_ref,
        out_shape_rc=he_ref_shape_rc,
    )

    marker_rows = []
    overlay_entries = []
    first_marker = True

    for marker in registrar.ihc_slides:
        ihc_img = registrar.ihc_padded[marker]
        rigid_reg = registrar.rigid_registrars[marker]
        nr_reg = registrar.non_rigid_registrars[marker]
        reg_shape_rc = registrar.reg_shape_rc[marker]

        ihc_color = warp_tools.warp_img(
            registrar.ihc_imgs[marker],
            M=registrar.ihc_padding_matrix[marker],
            out_shape_rc=reg_shape_rc,
        )

        if first_marker:
            fig = make_overlay_figure(he_img, ihc_img, "Unregistered")
            overlay_entries.append(
                {
                    "title": "Unregistered",
                    "data_uri": fig_to_data_uri(
                        fig, fmt=overlay_fmt, dpi=overlay_dpi, quality=overlay_quality
                    ),
                }
            )

            fig = make_rigid_overlay_figure(he_img, ihc_img, rigid_reg.M, "Rigid")
            overlay_entries.append(
                {
                    "title": "Rigid",
                    "data_uri": fig_to_data_uri(
                        fig, fmt=overlay_fmt, dpi=overlay_dpi, quality=overlay_quality
                    ),
                }
            )

            fig = make_non_rigid_overlay_figure(he_img, ihc_img, nr_reg, "Non-rigid")
            overlay_entries.append(
                {
                    "title": "Non-rigid",
                    "data_uri": fig_to_data_uri(
                        fig, fmt=overlay_fmt, dpi=overlay_dpi, quality=overlay_quality
                    ),
                }
            )
            first_marker = False

        fig = make_single_thumbnail_figure(he_color_ref, "HE")
        he_thumb_uri = fig_to_data_uri(
            fig, fmt=thumb_fmt, dpi=thumb_dpi, quality=thumb_quality
        )

        warped_ihc_color = nr_reg.warp_image(ihc_color, out_shape_rc=he_ref_shape_rc)
        fig = make_single_thumbnail_figure(warped_ihc_color, f"{marker} registered")
        ihc_thumb_uri = fig_to_data_uri(
            fig, fmt=thumb_fmt, dpi=thumb_dpi, quality=thumb_quality
        )

        if nr_reg.bk_dxdy is not None:
            fig = make_deformation_field_figure(
                nr_reg.bk_dxdy, f"{marker} deformation field"
            )
            def_uri = fig_to_data_uri(
                fig, fmt=def_fmt, dpi=def_dpi, quality=def_quality
            )
        else:
            def_uri = ""

        metrics = compute_marker_metrics(
            rigid_registrar=rigid_reg,
            non_rigid_registrar=nr_reg,
            mpp=mpp_canvas,
        )

        marker_rows.append(
            {
                "marker": marker,
                "original_displacement_um": metrics["original_displacement_um"],
                "rigid_displacement_um": metrics["rigid_displacement_um"],
                "non_rigid_displacement_um": metrics["non_rigid_displacement_um"],
                "rtre": metrics["rtre"],
                "n_matches": metrics["n_matches"],
                "he_thumb_uri": he_thumb_uri,
                "ihc_thumb_uri": ihc_thumb_uri,
                "def_uri": def_uri,
            }
        )

    overall_metrics = compute_overall_metrics(
        {row["marker"]: row for row in marker_rows}
    )

    he_ref_thumb_uri = fig_to_data_uri(
        make_single_thumbnail_figure(he_color_ref, "HE reference"),
        fmt=thumb_fmt,
        dpi=thumb_dpi,
        quality=thumb_quality,
    )

    return {
        "overall_metrics": overall_metrics,
        "overlay_entries": overlay_entries,
        "marker_rows": marker_rows,
        "he_ref_thumb_uri": he_ref_thumb_uri,
    }


def _build_hisalign_kwargs(
    config: dict, mpp_override: float | None = None
) -> dict[str, Any]:
    """Build kwargs for ``HisAlign`` from config and CLI overrides."""
    kwargs = {
        "registration_level": config.get("registration_level", 3),
        "max_image_dim_px": config.get("max_image_dim_px", 1024),
        "preprocessing": config.get("preprocessing", "od"),
        "feature_detector": config.get("feature_detector", "kaze"),
        "feature_n_levels": config.get("feature_n_levels", 3),
        "match_max_ratio": config.get("match_max_ratio", 1.0),
    }
    if mpp_override is not None:
        kwargs["mpp"] = mpp_override
    elif config.get("mpp") is not None:
        kwargs["mpp"] = config["mpp"]
    return kwargs


def _cmd_register(args: argparse.Namespace) -> None:
    """Handle the ``register`` subcommand."""
    config = _load_config(args.config)
    ihc_paths = _parse_ihc_args(args.ihc)
    he_path = Path(args.he).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    slide_id = _slide_id_from_he(he_path)
    kwargs = _build_hisalign_kwargs(config, mpp_override=args.mpp)

    aligner = HisAlign(he_path=he_path, ihc_paths=ihc_paths, **kwargs)
    model = aligner.fit()
    model.save(output_path)
    logger.info("Saved registration model to %s", output_path)

    if config.get("viz_sample_n", 0) > 0 or config.get("generate_report", True):
        output_dir = output_path.parent
        _generate_visualizations(
            registrar=aligner._registrar,
            he_path=he_path,
            ihc_paths=ihc_paths,
            config=config,
            output_dir=output_dir,
            slide_id=slide_id,
        )


def _cmd_warp(args: argparse.Namespace) -> None:
    """Handle the ``warp`` subcommand."""
    model = HisAlignModel.load(args.model)
    coords_df = pd.read_csv(args.coords)
    if "x" not in coords_df.columns or "y" not in coords_df.columns:
        raise ValueError("Coords CSV must contain 'x' and 'y' columns")

    coords = coords_df[["x", "y"]].to_numpy(dtype=np.float64)
    mapped = model.warp_xy(coords, marker=args.marker, direction=args.direction)

    out_df = pd.DataFrame(
        {
            "x": mapped[:, 0],
            "y": mapped[:, 1],
            "marker": args.marker,
            "direction": args.direction,
        }
    )
    out_df.to_csv(args.output, index=False)
    logger.info("Wrote %d mapped coordinates to %s", len(out_df), args.output)


def _cmd_visualize(args: argparse.Namespace) -> None:
    """Handle the ``visualize`` subcommand.

    The model file provides paths and configuration; registration is re-run to
    obtain the internal registrar objects needed for report/gallery generation.
    """
    model = HisAlignModel.load(args.model)
    config = _load_config(args.config)
    # Allow config from CLI to override stored config values
    merged_config = {**model.config, **config}

    he_path = Path(model.he_path)
    ihc_paths = {k: Path(v) for k, v in model.ihc_paths.items()}
    slide_id = _slide_id_from_he(he_path)

    kwargs = _build_hisalign_kwargs(merged_config)
    aligner = HisAlign(he_path=he_path, ihc_paths=ihc_paths, **kwargs)
    aligner.fit()

    output_dir = Path(args.output_dir)
    _generate_visualizations(
        registrar=aligner._registrar,
        he_path=he_path,
        ihc_paths=ihc_paths,
        config=merged_config,
        output_dir=output_dir,
        slide_id=slide_id,
    )


def main() -> None:
    """CLI entry point for hisalign."""
    parser = argparse.ArgumentParser(
        prog="hisalign",
        description="HISAlign: whole-slide image alignment for H&E and IHC markers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- register ---
    register_parser = subparsers.add_parser(
        "register", help="Register HE to IHC slides"
    )
    register_parser.add_argument(
        "--he", required=True, type=Path, help="HE reference slide path"
    )
    register_parser.add_argument(
        "--ihc",
        required=True,
        action="append",
        help="IHC slide path, optionally with marker name (e.g. CD3=CD3.svs or just CD3.svs)",
    )
    register_parser.add_argument(
        "--output", required=True, type=Path, help="Output .pkl model path"
    )
    register_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Config YAML (default: configs/default.yaml)",
    )
    register_parser.add_argument(
        "--mpp", type=float, default=None, help="Override MPP in µm/px"
    )
    register_parser.set_defaults(func=_cmd_register)

    # --- warp ---
    warp_parser = subparsers.add_parser(
        "warp", help="Warp coordinates using a saved model"
    )
    warp_parser.add_argument(
        "--model", required=True, type=Path, help="Input .pkl model"
    )
    warp_parser.add_argument("--marker", required=True, type=str, help="Marker name")
    warp_parser.add_argument(
        "--direction",
        required=True,
        choices=["he_to_ihc", "ihc_to_he"],
        help="Mapping direction",
    )
    warp_parser.add_argument(
        "--coords", required=True, type=Path, help="CSV with x,y columns"
    )
    warp_parser.add_argument(
        "--output", required=True, type=Path, help="Output CSV path"
    )
    warp_parser.set_defaults(func=_cmd_warp)

    # --- visualize ---
    viz_parser = subparsers.add_parser("visualize", help="Generate gallery and report")
    viz_parser.add_argument(
        "--model", required=True, type=Path, help="Input .pkl model"
    )
    viz_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config YAML to override stored config",
    )
    viz_parser.add_argument(
        "--output-dir", required=True, type=Path, help="Output directory"
    )
    viz_parser.set_defaults(func=_cmd_visualize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

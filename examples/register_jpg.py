"""Minimal example: register two JPG/PNG images with hisalign.

This example shows how to use ``HisAlign`` with ordinary raster images instead
of whole-slide formats. Because a JPG/PNG has only one resolution level, set
``registration_level=0`` and provide an explicit ``mpp`` value.

Usage with your own images
--------------------------

    python examples/register_jpg.py \
        --he examples/data/he.jpg \
        --ihc examples/data/ihc.jpg \
        --output-dir ./out

Run on synthetic data
---------------------

    python examples/register_jpg.py --synthetic --output-dir ./out

Outputs
-------

- ``out/model.pkl``            -- serializable HisAlignModel
- ``out/he.jpg`` / ``ihc.jpg`` -- input images (when --synthetic)
- ``out/00_unregistered.png``  -- green/magenta overlay before registration
- ``out/01_rigid.png``         -- overlay after rigid registration
- ``out/02_nonrigid.png``      -- overlay after non-rigid registration
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from hisalign import HisAlign
from hisalign.viz import (
    make_non_rigid_overlay_figure,
    make_overlay_figure,
    make_rigid_overlay_figure,
)


def _save_fig(fig, path: Path) -> None:
    """Save a matplotlib figure and close it."""
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _generate_synthetic_images(output_dir: Path, size: int = 512) -> tuple[Path, Path]:
    """Create a deterministic HE image and a transformed IHC image."""
    rng = np.random.default_rng(42)

    # Base pattern with lots of edges/textures for feature detection
    base = np.full((size, size, 3), 240, dtype=np.uint8)
    for _ in range(80):
        color = tuple(int(c) for c in rng.integers(30, 220, size=3))
        center = tuple(int(c) for c in rng.integers(0, size, size=2))
        radius = int(rng.integers(5, 40))
        cv2.circle(base, center, radius, color, -1)

    for i in range(0, size, 32):
        cv2.line(base, (i, 0), (i, size), (180, 180, 180), 1)
        cv2.line(base, (0, i), (size, i), (180, 180, 180), 1)

    he_path = output_dir / "he.jpg"
    Image.fromarray(base).save(he_path, quality=95)

    # Apply a small translation + rotation to create the moving image
    angle = 3.0
    tx, ty = 12.0, -8.0
    center = (size // 2, size // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    ihc = cv2.warpAffine(base, matrix, (size, size), borderValue=(240, 240, 240))

    ihc_path = output_dir / "ihc.jpg"
    Image.fromarray(ihc).save(ihc_path, quality=95)

    return he_path, ihc_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register two JPG/PNG images with hisalign."
    )
    parser.add_argument(
        "--he",
        type=Path,
        help="Path to the HE reference image (required unless --synthetic).",
    )
    parser.add_argument(
        "--ihc",
        type=Path,
        help="Path to the moving IHC image (required unless --synthetic).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./out"),
        help="Directory for the saved model and overlay images.",
    )
    parser.add_argument(
        "--registration-level",
        type=int,
        default=0,
        help="Pyramid level to use (0 for static images).",
    )
    parser.add_argument(
        "--max-image-dim-px",
        type=int,
        default=1024,
        help="Maximum dimension of the processed registration image.",
    )
    parser.add_argument(
        "--mpp",
        type=float,
        default=1.0,
        help="Microns per pixel (static images have no slide metadata).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic HE/IHC images and register them.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        he_path, ihc_path = _generate_synthetic_images(args.output_dir)
    else:
        if not args.he or not args.ihc:
            parser.error("--he and --ihc are required unless using --synthetic")
        he_path = args.he
        ihc_path = args.ihc

    marker = ihc_path.stem.split()[-1] if ihc_path.stem else "IHC"

    print(f"HE : {he_path}")
    print(f"IHC: {ihc_path} (marker={marker})")

    aligner = HisAlign(
        he_path=he_path,
        ihc_paths=[ihc_path],
        registration_level=args.registration_level,
        max_image_dim_px=args.max_image_dim_px,
        preprocessing="od",
        feature_detector="kaze",
        feature_n_levels=3,
        match_max_ratio=1.0,
        mpp=args.mpp,
    )
    model = aligner.fit()

    model_path = args.output_dir / "model.pkl"
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # Visualize registration quality
    registrar = aligner._registrar
    he_img = registrar.he_padded
    ihc_img = registrar.ihc_padded[marker]
    rigid_reg = registrar.rigid_registrars[marker]
    nr_reg = registrar.non_rigid_registrars[marker]

    _save_fig(
        make_overlay_figure(he_img, ihc_img, "Unregistered"),
        args.output_dir / "00_unregistered.png",
    )
    _save_fig(
        make_rigid_overlay_figure(he_img, ihc_img, rigid_reg.M, "Rigid"),
        args.output_dir / "01_rigid.png",
    )
    _save_fig(
        make_non_rigid_overlay_figure(he_img, ihc_img, nr_reg, "Non-rigid"),
        args.output_dir / "02_nonrigid.png",
    )

    # Quick offline coordinate mapping sanity check
    coords = np.array([[50.0, 50.0]])
    mapped = model.warp_xy(coords, marker=marker, direction="he_to_ihc")
    print(f"Sample warp HE -> IHC: {coords[0]} -> {mapped[0]}")

    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()

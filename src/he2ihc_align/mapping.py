"""Build CSV-style DataFrame mapping each HE patch to each IHC marker."""

from __future__ import annotations

import numpy as np
import pandas as pd

from he2ihc_align.registration.registrar import HEIHCRegistrar
from he2ihc_align.slide_io.base import Slide


def build_mapping_table(
    registrar: HEIHCRegistrar,
    he_slide: Slide,
    ihc_slides: dict[str, Slide],
    he_patch_bboxes: list[tuple[int, int, int, int]],
    slide_id: str,
    he_level: int = 0,
    ihc_level: int = 0,
) -> pd.DataFrame:
    """Build a DataFrame mapping each HE patch to each IHC marker.

    For each HE patch, the four corners are mapped through the registration
    warp to IHC coordinates. The enclosing axis-aligned bounding box in IHC
    level-0 coordinates is computed.

    Parameters
    ----------
    registrar : HEIHCRegistrar
        Fitted registrar providing ``warp_xy_from_he_to_ihc``.
    he_slide : Slide
        HE reference slide (used for dimensions).
    ihc_slides : dict[str, Slide]
        Dictionary of IHC slides keyed by marker name.
    he_patch_bboxes : list[tuple[int, int, int, int]]
        HE patch bboxes in level-0 coordinates as (x, y, w, h).
    slide_id : str
        Identifier for the slide/case.
    he_level : int, default 0
        Pyramid level of the HE bboxes. Reserved for future use; must be 0.
    ihc_level : int, default 0
        Pyramid level of the IHC bboxes. Reserved for future use; must be 0.

    Raises
    ------
    ValueError
        If ``he_level`` or ``ihc_level`` is non-zero.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        patch_id, slide_id, marker, he_x, he_y, he_w, he_h,
        ihc_x, ihc_y, ihc_w, ihc_h, clipped.
    """
    if he_level != 0 or ihc_level != 0:
        raise ValueError("he_level and ihc_level must be 0 (reserved for future use)")

    rows: list[dict] = []

    for patch_idx, (he_x, he_y, he_w, he_h) in enumerate(he_patch_bboxes):
        patch_id = f"{slide_id}_{patch_idx:04d}"

        corners = _bbox_to_corners(he_x, he_y, he_w, he_h)

        for marker, ihc_slide in ihc_slides.items():
            # Warp corners from HE to IHC
            ihc_corners = registrar.warp_xy_from_he_to_ihc(corners, marker=marker)

            # Compute enclosing axis-aligned bbox
            ihc_x_min = int(np.floor(ihc_corners[:, 0].min()))
            ihc_y_min = int(np.floor(ihc_corners[:, 1].min()))
            ihc_x_max = int(np.ceil(ihc_corners[:, 0].max()))
            ihc_y_max = int(np.ceil(ihc_corners[:, 1].max()))

            ihc_x = ihc_x_min
            ihc_y = ihc_y_min
            ihc_w = ihc_x_max - ihc_x_min
            ihc_h = ihc_y_max - ihc_y_min

            # Check if bbox exceeds IHC slide bounds or is degenerate
            ihc_w_total, ihc_h_total = ihc_slide.level_dimensions[ihc_level]
            clipped = (
                ihc_w <= 0
                or ihc_h <= 0
                or ihc_x < 0
                or ihc_y < 0
                or ihc_x + ihc_w > ihc_w_total
                or ihc_y + ihc_h > ihc_h_total
            )

            rows.append(
                {
                    "patch_id": patch_id,
                    "slide_id": slide_id,
                    "marker": marker,
                    "he_x": he_x,
                    "he_y": he_y,
                    "he_w": he_w,
                    "he_h": he_h,
                    "ihc_x": ihc_x,
                    "ihc_y": ihc_y,
                    "ihc_w": ihc_w,
                    "ihc_h": ihc_h,
                    "clipped": clipped,
                }
            )

    return pd.DataFrame(rows)


def _bbox_to_corners(x: int, y: int, w: int, h: int) -> np.ndarray:
    """Return the four corners of a bounding box as a (4, 2) float array."""
    return np.array(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ],
        dtype=np.float64,
    )

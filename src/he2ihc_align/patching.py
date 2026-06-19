"""Grid patch sampling on HE slides with white/blank filtering."""

from __future__ import annotations

import numpy as np

from he2ihc_align.slide_io.base import Slide


def sample_grid_patches(
    slide: Slide,
    patch_size: int,
    stride: int | None = None,
    level: int = 0,
    margin: int = 0,
    white_threshold: int = 230,
    max_white_ratio: float = 0.95,
) -> list[tuple[int, int, int, int]]:
    """Sample grid patches from a whole-slide image, filtering out white/blank patches.

    Parameters
    ----------
    slide : Slide
        The slide to sample patches from.
    patch_size : int
        Width and height of each patch in pixels at the target level.
    stride : int, optional
        Step size between adjacent patches. Defaults to patch_size (no overlap).
    level : int, default 0
        Pyramid level to read patches from. Coordinates in the returned bboxes
        are always in level-0 pixel coordinates.
    margin : int, default 0
        Number of pixels to exclude from each edge of the slide in level-0 coordinates.
    white_threshold : int, default 230
        Grayscale intensity threshold above which a pixel is considered "white".
    max_white_ratio : float, default 0.95
        Maximum allowed ratio of white pixels in a patch. Patches exceeding this
        ratio are discarded.

    Returns
    -------
    list[tuple[int, int, int, int]]
        List of bounding boxes in level-0 coordinates as (x, y, w, h).
    """
    if stride is None:
        stride = patch_size

    level_w, level_h = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]

    # Apply margin in level-0 coordinates, then convert to level coordinates
    margin_level = int(margin / downsample)
    effective_w = max(0, level_w - 2 * margin_level)
    effective_h = max(0, level_h - 2 * margin_level)

    bboxes: list[tuple[int, int, int, int]] = []

    for y_level in range(margin_level, margin_level + effective_h, stride):
        for x_level in range(margin_level, margin_level + effective_w, stride):
            # Clamp patch to effective bounds at this level
            pw = min(patch_size, margin_level + effective_w - x_level)
            ph = min(patch_size, margin_level + effective_h - y_level)
            if pw <= 0 or ph <= 0:
                continue

            # Convert location to level-0 coordinates for read_region
            x0 = int(x_level * downsample)
            y0 = int(y_level * downsample)
            w0 = int(pw * downsample)
            h0 = int(ph * downsample)

            # Read patch at the specified level (location is level-0, size is at level)
            patch = slide.read_region((x0, y0), level, (pw, ph))

            # Convert to grayscale and count white pixels
            gray = np.mean(patch, axis=2) if patch.ndim == 3 else patch
            white_pixels = np.sum(gray > white_threshold)
            total_pixels = gray.size
            white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0.0

            if white_ratio > max_white_ratio:
                continue

            bboxes.append((x0, y0, w0, h0))

    return bboxes

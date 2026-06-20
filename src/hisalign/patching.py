"""Grid patch sampling on HE slides with white/blank filtering."""

from __future__ import annotations

import numpy as np

from hisalign.slide_io.base import Slide


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

    # Build list of all patch positions before reading
    patch_positions: list[tuple[int, int, int, int]] = []
    for y_level in range(margin_level, margin_level + effective_h, stride):
        for x_level in range(margin_level, margin_level + effective_w, stride):
            pw = min(patch_size, margin_level + effective_w - x_level)
            ph = min(patch_size, margin_level + effective_h - y_level)
            if pw <= 0 or ph <= 0:
                continue
            x0 = int(x_level * downsample)
            y0 = int(y_level * downsample)
            w0 = int(pw * downsample)
            h0 = int(ph * downsample)
            patch_positions.append((x_level, y_level, pw, ph, x0, y0, w0, h0))

    if not patch_positions:
        return bboxes

    # Determine tile size to batch read_region calls
    tile_size = max(patch_size, stride) * 4

    # Group patches by tile
    tile_patches: dict[
        tuple[int, int], list[tuple[int, int, int, int, int, int, int, int]]
    ] = {}
    for x_level, y_level, pw, ph, x0, y0, w0, h0 in patch_positions:
        tile_x = (x_level // tile_size) * tile_size
        tile_y = (y_level // tile_size) * tile_size
        tile_patches.setdefault((tile_x, tile_y), []).append(
            (x_level, y_level, pw, ph, x0, y0, w0, h0)
        )

    for (tile_x, tile_y), patches in tile_patches.items():
        # Compute tile bounds covering all patches in this tile
        max_x = max(x_level + pw for x_level, _, pw, _, _, _, _, _ in patches)
        max_y = max(y_level + ph for _, y_level, _, ph, _, _, _, _ in patches)
        tile_w = max_x - tile_x
        tile_h = max_y - tile_y

        tile_x0 = int(tile_x * downsample)
        tile_y0 = int(tile_y * downsample)
        tile = slide.read_region((tile_x0, tile_y0), level, (tile_w, tile_h))

        for x_level, y_level, pw, ph, x0, y0, w0, h0 in patches:
            rel_x = x_level - tile_x
            rel_y = y_level - tile_y
            patch = tile[rel_y : rel_y + ph, rel_x : rel_x + pw]

            gray = np.mean(patch, axis=2) if patch.ndim == 3 else patch
            white_pixels = np.sum(gray > white_threshold)
            total_pixels = gray.size
            white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0.0

            if white_ratio > max_white_ratio:
                continue

            bboxes.append((x0, y0, w0, h0))

    return bboxes

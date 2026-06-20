"""Preprocessing utilities for whole-slide image registration.

Imitates VALIS brightfield preprocessing without depending on VALIS.
"""

from __future__ import annotations

import numpy as np
from skimage import exposure, morphology
from skimage.filters.rank import entropy
from skimage.morphology import disk


def optical_density_gray(image: np.ndarray, p: int = 95) -> np.ndarray:
    """Convert an RGB image to optical-density norm grayscale.

    Parameters
    ----------
    image : np.ndarray
        HWC uint8 RGB image.
    p : int
        Upper percentile used for clipping (default 95).

    Returns
    -------
    np.ndarray
        2D uint8 grayscale image.
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    img01 = image.astype(np.float64) / 255.0
    eps = np.finfo(float).eps
    od = -np.log10(img01 + eps)
    od_norm = np.mean(od, axis=2)

    upper_p = np.percentile(od_norm, p)
    od_clipped = np.clip(od_norm, 0.0, upper_p)

    processed = exposure.rescale_intensity(od_clipped, out_range=np.uint8)
    return processed.astype(np.uint8)


def tissue_mask(image: np.ndarray, entropy_radius: int = 3) -> np.ndarray:
    """Create a rough tissue mask from an RGB/gray image.

    Uses local entropy on a grayscale version of the image.  This is a
    lightweight approximation of VALIS's entropy-based masking and is intended
    to focus feature detection on tissue regions.

    Parameters
    ----------
    image : np.ndarray
        HWC uint8 RGB or 2D grayscale image.
    entropy_radius : int
        Radius of the entropy disk.

    Returns
    -------
    np.ndarray
        Binary uint8 mask (255 = tissue, 0 = background).
    """
    if image.ndim == 3:
        gray = np.mean(image.astype(np.float64), axis=2)
    else:
        gray = image.astype(np.float64)

    gray = exposure.rescale_intensity(gray, out_range=(0, 255)).astype(np.uint8)

    ent = entropy(gray, disk(entropy_radius))
    # Otsu threshold on entropy; tissue has higher entropy than empty background
    try:
        from skimage.filters import threshold_otsu

        thresh = threshold_otsu(ent)
    except Exception:
        thresh = np.percentile(ent, 50)

    mask = (ent > thresh).astype(np.uint8) * 255

    # Clean small specks
    mask = morphology.remove_small_objects(mask.astype(bool)).astype(np.uint8) * 255

    return mask


def mask_to_bbox_mask(mask: np.ndarray) -> np.ndarray:
    """Expand a mask to cover the bounding boxes of its connected components.

    This mirrors VALIS's `mask2bbox_mask` utility: each connected foreground
    region is replaced by its axis-aligned bounding box.

    Parameters
    ----------
    mask : np.ndarray
        Binary uint8 mask.

    Returns
    -------
    np.ndarray
        Binary uint8 mask where each component's bbox is filled.
    """
    from skimage.measure import label, regionprops

    labeled = label(mask > 0)
    bbox_mask = np.zeros_like(mask)
    for region in regionprops(labeled):
        min_r, min_c, max_r, max_c = region.bbox
        bbox_mask[min_r:max_r, min_c:max_c] = 255

    return bbox_mask

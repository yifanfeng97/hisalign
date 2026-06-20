"""Slide factory."""

from __future__ import annotations

from pathlib import Path

import openslide

from hisalign.slide_io.base import Slide
from hisalign.slide_io.image_backend import ImageSlideBackend
from hisalign.slide_io.kfb_backend import KfbSlideBackend
from hisalign.slide_io.openslide_backend import OpenSlideBackend

# Extensions natively supported by OpenSlide
OPENSLIDE_EXTS = {".svs", ".tif", ".tiff", ".mrxs", ".ndpi", ".scn", ".vms", ".vmu"}

# Common static raster image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def open_slide(path: str | Path) -> Slide:
    """Open a slide file and return the appropriate backend.

    Args:
        path: Path to the slide file.

    Returns:
        A slide backend instance (OpenSlideBackend, KfbSlideBackend, or
        ImageSlideBackend).

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".kfb":
        return KfbSlideBackend(path)
    if ext in OPENSLIDE_EXTS:
        return OpenSlideBackend(path)
    if ext in IMAGE_EXTS:
        return ImageSlideBackend(path)
    raise ValueError(f"Unsupported slide format: {ext}")


def get_slide_mpp(slide: Slide) -> float | None:
    """Return the microns-per-pixel value for a slide, if available.

    Tries the OpenSlide MPP_X property first, then common vendor-specific keys.
    Returns ``None`` when no MPP metadata is present (typical for KFBio slides).
    """
    props = slide.properties
    keys = [
        openslide.PROPERTY_NAME_MPP_X,
        "tiff.XResolution",
        "openslide.mpp-x",
        "aperio.MPP",
    ]
    for key in keys:
        value = props.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (ValueError, TypeError):
            continue
    return None

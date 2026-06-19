"""Slide factory."""

from __future__ import annotations

from pathlib import Path

from he2ihc_align.slide_io.base import Slide
from he2ihc_align.slide_io.openslide_backend import OpenSlideBackend
from he2ihc_align.slide_io.kfb_backend import KfbSlideBackend


# Extensions natively supported by OpenSlide
OPENSLIDE_EXTS = {".svs", ".tif", ".tiff", ".mrxs", ".ndpi", ".scn", ".vms", ".vmu"}


def open_slide(path: str | Path) -> Slide:
    """Open a slide file and return the appropriate backend.

    Args:
        path: Path to the slide file.

    Returns:
        A slide backend instance (OpenSlideBackend or KfbSlideBackend).

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".kfb":
        return KfbSlideBackend(path)
    elif ext in OPENSLIDE_EXTS:
        return OpenSlideBackend(path)
    else:
        raise ValueError(f"Unsupported slide format: {ext}")

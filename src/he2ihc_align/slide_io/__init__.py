"""Slide I/O package."""

from he2ihc_align.slide_io.base import Slide, SlideIOError
from he2ihc_align.slide_io.factory import open_slide
from he2ihc_align.slide_io.kfb_backend import KfbSlideBackend
from he2ihc_align.slide_io.openslide_backend import OpenSlideBackend

__all__ = [
    "Slide",
    "SlideIOError",
    "open_slide",
    "KfbSlideBackend",
    "OpenSlideBackend",
]

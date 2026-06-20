"""Slide I/O package."""

from hisalign.slide_io.base import Slide, SlideIOError
from hisalign.slide_io.factory import open_slide
from hisalign.slide_io.kfb_backend import KfbSlideBackend
from hisalign.slide_io.openslide_backend import OpenSlideBackend

__all__ = [
    "Slide",
    "SlideIOError",
    "open_slide",
    "KfbSlideBackend",
    "OpenSlideBackend",
]

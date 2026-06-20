"""Backend for static raster images (JPG/PNG/BMP) using PIL.

This lets ``HisAlign`` and the CLI work with ordinary images in addition to
whole-slide formats. Because a static image has only one resolution level,
``registration_level`` should be set to ``0``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from hisalign.slide_io.base import SlideIOError, _pil_to_rgb_array


class ImageSlideBackend:
    """Slide-like wrapper around a single PIL-compatible raster image."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._pil = Image.open(self._path)
        self._img = _pil_to_rgb_array(self._pil)

    @property
    def level_count(self) -> int:
        return 1

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        w, h = self._img.shape[1], self._img.shape[0]
        return [(w, h)]

    @property
    def level_downsamples(self) -> list[float]:
        return [1.0]

    @property
    def properties(self) -> dict[str, str]:
        return {}

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> np.ndarray:
        """Return a HWC uint8 RGB crop from the image.

        Both ``location`` and ``size`` are interpreted in level-0 pixel
        coordinates, which is the only level available for a static image.
        Requests are clamped to the image bounds.
        """
        if level != 0:
            raise SlideIOError(
                f"ImageSlideBackend only supports level 0, got {level}"
            )

        x, y = location
        w, h = size
        img_h, img_w = self._img.shape[:2]

        x = max(0, min(x, img_w))
        y = max(0, min(y, img_h))
        w = max(0, min(w, img_w - x))
        h = max(0, min(h, img_h - y))

        if w == 0 or h == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)

        return self._img[y : y + h, x : x + w].copy()

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return 0

    def close(self) -> None:
        self._pil.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"ImageSlideBackend(path={str(self._path)!r}, "
            f"dimensions={self.level_dimensions[0]!r})"
        )

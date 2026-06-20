"""Slide I/O base protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image


class SlideIOError(Exception):
    """Raised when a slide I/O operation fails."""


def _pil_to_rgb_array(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a HWC uint8 RGB numpy array.

    Handles RGBA images by compositing onto a white background.
    """
    if pil_img.mode == "RGBA":
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_img = bg
    else:
        pil_img = pil_img.convert("RGB")
    return np.array(pil_img)


@runtime_checkable
class Slide(Protocol):
    """Protocol for whole-slide image readers."""

    @property
    def level_count(self) -> int:
        """Number of resolution levels."""
        ...

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        """Dimensions (width, height) for each level."""
        ...

    @property
    def level_downsamples(self) -> list[float]:
        """Downsample factor for each level relative to level 0."""
        ...

    @property
    def properties(self) -> dict[str, str]:
        """Slide metadata as key-value strings."""
        ...

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> np.ndarray:
        """Read a region from the slide.

        Args:
            location: (x, y) top-left corner in level-0 coordinates.
            level: Pyramid level to read from.
            size: (width, height) of the region to read.

        Returns:
            HWC uint8 RGB numpy array.
        """
        ...

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Return the best pyramid level for a given downsample factor."""
        ...

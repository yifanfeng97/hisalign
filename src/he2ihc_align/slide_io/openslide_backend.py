"""OpenSlide backend implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import openslide


class OpenSlideBackend:
    """Wrapper around openslide-python exposing the Slide protocol."""

    def __init__(self, path: str | Path) -> None:
        self._slide = openslide.OpenSlide(str(path))

    @property
    def level_count(self) -> int:
        return self._slide.level_count

    @property
    def level_dimensions(self) -> list[Tuple[int, int]]:
        return list(self._slide.level_dimensions)

    @property
    def level_downsamples(self) -> list[float]:
        return list(self._slide.level_downsamples)

    @property
    def properties(self) -> dict[str, str]:
        return dict(self._slide.properties)

    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
        """Return HWC uint8 RGB numpy array."""
        pil_img = self._slide.read_region(location, level, size)
        # Convert RGBA to RGB
        if pil_img.mode == "RGBA":
            # Create a white background
            from PIL import Image
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[3])
            pil_img = bg
        else:
            pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img)
        return arr

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return self._slide.get_best_level_for_downsample(downsample)

    def close(self) -> None:
        self._slide.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

"""KfbSlide backend implementation."""

from __future__ import annotations

from pathlib import Path

import kfbslide
import numpy as np

from he2ihc_align.slide_io.base import SlideIOError, _pil_to_rgb_array


class KfbSlideBackend:
    """Wrapper around kfbslide exposing the Slide protocol."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._slide = kfbslide.KfbSlide(str(path))

    @property
    def level_count(self) -> int:
        return self._slide.level_count

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        return list(self._slide.level_dimensions)

    @property
    def level_downsamples(self) -> list[float]:
        return list(self._slide.level_downsamples)

    @property
    def properties(self) -> dict[str, str]:
        return dict(self._slide.properties)

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> np.ndarray:
        """Return HWC uint8 RGB numpy array.

        ``location`` is in level-0 pixel coordinates (same as OpenSlide).
        ``size`` is in target-level pixel coordinates.
        """
        if level < 0 or level >= self.level_count:
            raise SlideIOError(f"Invalid level {level}: must be between 0 and {self.level_count - 1}")
        x, y = location
        w, h = size
        downsample = self.level_downsamples[level]
        level0_w, level0_h = self.level_dimensions[0]
        if x < 0 or y < 0 or x + w * downsample > level0_w + downsample or y + h * downsample > level0_h + downsample:
            raise SlideIOError(
                f"Region out of bounds: location {location}, size {size}, level {level} "
                f"exceeds level-0 dimensions {self.level_dimensions[0]}"
            )
        location_level = (int(x / downsample), int(y / downsample))
        try:
            pil_img = self._slide.read_region(location_level, level, size)
        except kfbslide.OpenSlideError as exc:
            raise SlideIOError(
                f"KfbSlide failed to read region at {location}, level {level}, size {size}: {exc}"
            ) from exc
        return _pil_to_rgb_array(pil_img)

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return self._slide.get_best_level_for_downsample(downsample)

    def close(self) -> None:
        self._slide.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        return f"KfbSlideBackend(path={str(self._path)!r}, level_count={self.level_count})"

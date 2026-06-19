"""Slide I/O base protocol."""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class Slide(Protocol):
    """Protocol for whole-slide image readers."""

    @property
    def level_count(self) -> int:
        """Number of resolution levels."""
        ...

    @property
    def level_dimensions(self) -> list[Tuple[int, int]]:
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

    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
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

"""Tests for preprocessing utilities."""

from __future__ import annotations

import numpy as np

from hisalign import preprocessing


def test_optical_density_gray_shape():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    gray = preprocessing.optical_density_gray(rgb)
    assert gray.shape == (64, 64)
    assert gray.dtype == np.uint8


def test_optical_density_gray_increases_contrast_for_he_like():
    # Synthetic HE-like: mostly white background with a purple-ish tissue blob
    rgb = np.full((64, 64, 3), 240, dtype=np.uint8)
    rgb[16:48, 16:48] = [160, 100, 180]
    gray = preprocessing.optical_density_gray(rgb)
    # In OD norm, tissue (darker input) has higher OD and becomes brighter
    # after rescaling, while white background becomes dark.
    assert np.mean(gray[16:48, 16:48]) > np.mean(gray[:16, :16])


def test_tissue_mask_returns_binary_mask():
    rgb = np.full((64, 64, 3), 240, dtype=np.uint8)
    rgb[16:48, 16:48] = [160, 100, 180]
    mask = preprocessing.tissue_mask(rgb)
    assert mask.shape == (64, 64)
    assert set(np.unique(mask)).issubset({0, 255})


def test_mask_to_bbox_mask_fills_bounding_boxes():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:15, 10:15] = 255
    mask[50:52, 50:52] = 255
    bbox = preprocessing.mask_to_bbox_mask(mask)
    # The first component's bbox should be filled
    assert bbox[10:15, 10:15].all()
    assert bbox.sum() >= mask.sum()

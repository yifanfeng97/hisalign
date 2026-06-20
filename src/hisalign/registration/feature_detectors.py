"""Simplified feature detectors for HE-to-IHC registration.

Based on VALIS feature_detectors.py, trimmed to MVP.
Keeps only OpenCV-based detectors (no torch/kornia dependencies).
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage import exposure

MAX_FEATURES = 7500


def filter_features(kp, desc, n_keep=MAX_FEATURES):
    """Keep top features by response."""
    if len(kp) <= n_keep:
        return kp, desc
    response = np.array([x.response for x in kp])
    keep_idx = np.argsort(response)[::-1][0:n_keep]
    return [kp[i] for i in keep_idx], desc[keep_idx, :]


class FeatureDD:
    """Abstract base class for feature detection and description."""

    def __init__(self, kp_detector=None, kp_descriptor=None, rgb=False, n_levels=1):
        self.kp_detector = kp_detector
        self.kp_descriptor = kp_descriptor
        self.rgb = rgb
        self.n_levels = n_levels

        if kp_descriptor is not None and kp_detector is not None:
            self.kp_descriptor_name = kp_descriptor.__class__.__name__
            self.kp_detector_name = kp_detector.__class__.__name__
        elif kp_descriptor is None and kp_detector is not None:
            self.kp_descriptor_name = kp_detector.__class__.__name__
            self.kp_detector_name = self.kp_descriptor_name
        elif kp_descriptor is not None and kp_detector is None:
            self.kp_descriptor_name = kp_descriptor.__class__.__name__
            self.kp_detector_name = self.kp_descriptor_name

    def _detect_and_compute(self, image, mask=None):
        """Detect and compute features in a single image."""
        image = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
        if self.kp_detector is not None:
            detected_kp = self.kp_detector.detect(image)
            kp, desc = self.kp_descriptor.compute(image, detected_kp)
        else:
            kp, desc = self.kp_descriptor.detectAndCompute(image, mask=mask)

        if desc is None or len(kp) == 0:
            return np.zeros((0, 2)), np.zeros((0, 64))

        if desc.shape[0] > MAX_FEATURES:
            kp, desc = filter_features(kp, desc)

        kp_pos_xy = np.array([k.pt for k in kp])
        return kp_pos_xy, desc

    def detect_and_compute(self, image, mask=None):
        """Detect and compute features, optionally at multiple scales."""
        if self.n_levels <= 1:
            return self._detect_and_compute(image, mask)

        all_kp = []
        all_desc = []
        s = 0.5
        detect_img = image
        for i in range(self.n_levels):
            kp_pos_xy, desc = self._detect_and_compute(detect_img, mask)
            kp_pos_xy *= 1 / (s**i)
            all_kp.append(kp_pos_xy)
            all_desc.append(desc)
            detect_img = cv2.resize(detect_img, None, fx=s, fy=s)

        all_kp = np.vstack(all_kp)
        all_desc = np.vstack(all_desc)
        return all_kp, all_desc


class BriskFD(FeatureDD):
    """BRISK feature detector/descriptor."""

    def __init__(self, n_levels=1):
        brisk = cv2.BRISK_create()
        super().__init__(kp_detector=brisk, kp_descriptor=brisk, n_levels=n_levels)


class KazeFD(FeatureDD):
    """KAZE feature detector/descriptor."""

    def __init__(self, n_levels=1):
        kaze = cv2.KAZE_create()
        super().__init__(kp_detector=kaze, kp_descriptor=kaze, n_levels=n_levels)


class AkazeFD(FeatureDD):
    """AKAZE feature detector/descriptor."""

    def __init__(self, n_levels=1):
        akaze = cv2.AKAZE_create()
        super().__init__(kp_detector=akaze, kp_descriptor=akaze, n_levels=n_levels)


class OrbFD(FeatureDD):
    """ORB feature detector/descriptor."""

    def __init__(self, nfeatures=MAX_FEATURES, n_levels=1):
        orb = cv2.ORB_create(nfeatures=nfeatures)
        super().__init__(kp_detector=orb, kp_descriptor=orb, n_levels=n_levels)


class VggFD(FeatureDD):
    """VGG descriptor with ORB detector (good for matching).

    Note: Requires opencv-contrib-python. Falls back to ORB if not available.
    """

    def __init__(self, n_levels=1):
        orb = cv2.ORB_create(nfeatures=MAX_FEATURES)
        try:
            vgg = cv2.xfeatures2d.VGG_create()
            super().__init__(kp_detector=orb, kp_descriptor=vgg, n_levels=n_levels)
        except AttributeError:
            # xfeatures2d not available, use ORB only
            super().__init__(kp_detector=orb, n_levels=n_levels)


class SiftFD(FeatureDD):
    """SIFT feature detector/descriptor.

    SIFT is available in the main opencv-python package since the patent
    expired. Falls back to AKAZE if SIFT cannot be created.
    """

    def __init__(self, n_levels=1):
        try:
            sift = cv2.SIFT_create()
            super().__init__(kp_detector=sift, kp_descriptor=sift, n_levels=n_levels)
        except (AttributeError, cv2.error):
            # Older/non-free OpenCV builds may not expose SIFT
            akaze = cv2.AKAZE_create()
            super().__init__(kp_detector=akaze, kp_descriptor=akaze, n_levels=n_levels)


def create_feature_detector(name: str, n_levels: int = 1) -> FeatureDD:
    """Factory for creating a feature detector by name.

    Parameters
    ----------
    name : str
        One of ``brisk``, ``akaze``, ``kaze``, ``orb``, ``sift``, ``vgg``.
    n_levels : int
        Number of detection scales (passed to multi-scale detectors).

    Returns
    -------
    FeatureDD
        Configured feature detector.
    """
    name = name.lower().strip()
    mapping: dict[str, type[FeatureDD]] = {
        "brisk": BriskFD,
        "akaze": AkazeFD,
        "kaze": KazeFD,
        "orb": OrbFD,
        "sift": SiftFD,
        "vgg": VggFD,
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown feature detector '{name}'. Choose from {list(mapping.keys())}."
        )
    return mapping[name](n_levels=n_levels)

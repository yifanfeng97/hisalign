"""Simplified rigid registration for HE-to-IHC alignment.

Based on VALIS serial_rigid.py, trimmed to MVP.
Simplified to single reference + N moving images (no serial Z-stack logic).
"""

from __future__ import annotations

import numpy as np
import cv2
from skimage.transform import EuclideanTransform, SimilarityTransform, AffineTransform
from tqdm import tqdm

from . import warp_tools
from . import feature_detectors
from . import feature_matcher


class RigidRegistrar:
    """Rigid registration of a moving image to a reference image.

    Attributes
    ----------
    ref_img : ndarray
        Reference image (grayscale, uint8).
    moving_img : ndarray
        Moving image (grayscale, uint8).
    ref_name : str
        Name of reference image.
    moving_name : str
        Name of moving image.
    M : ndarray
        3x3 transformation matrix that warps moving to reference.
    matched_kp_ref : ndarray
        (N, 2) matched keypoints in reference image.
    matched_kp_moving : ndarray
        (N, 2) matched keypoints in moving image.
    """

    def __init__(self, ref_img, moving_img, ref_name="ref", moving_name="moving"):
        self.ref_img = ref_img
        self.moving_img = moving_img
        self.ref_name = ref_name
        self.moving_name = moving_name
        self.M = np.identity(3)
        self.matched_kp_ref = None
        self.matched_kp_moving = None
        self.n_matches = 0

    def fit(self, feature_detector=None, matcher=None, transform_type="similarity"):
        """Run rigid registration.

        Parameters
        ----------
        feature_detector : FeatureDD, optional
            Feature detector. Defaults to BriskFD.
        matcher : Matcher, optional
            Feature matcher. Defaults to Matcher with RANSAC.
        transform_type : str
            Type of transform: "euclidean", "similarity", or "affine".

        Returns
        -------
        self
        """
        if feature_detector is None:
            feature_detector = feature_detectors.BriskFD()
        if matcher is None:
            matcher = feature_matcher.Matcher()

        # Detect features
        ref_kp, ref_desc = feature_detector.detect_and_compute(self.ref_img)
        moving_kp, moving_desc = feature_detector.detect_and_compute(self.moving_img)

        if len(ref_kp) == 0 or len(moving_kp) == 0:
            print(f"Warning: No features found for {self.moving_name}")
            return self

        # Match features
        match_info12, filtered_match_info12, match_info21, filtered_match_info21 = \
            matcher.match_images(
                img1=self.ref_img, desc1=ref_desc, kp1_xy=ref_kp,
                img2=self.moving_img, desc2=moving_desc, kp2_xy=moving_kp,
            )

        self.matched_kp_ref = filtered_match_info12.matched_kp1_xy
        self.matched_kp_moving = filtered_match_info12.matched_kp2_xy
        self.n_matches = filtered_match_info12.n_matches

        if self.n_matches < 3:
            print(f"Warning: Only {self.n_matches} matches found for {self.moving_name}. Using identity transform.")
            return self

        # Estimate transform
        if transform_type == "euclidean":
            tform = EuclideanTransform()
        elif transform_type == "similarity":
            tform = SimilarityTransform()
        else:
            tform = AffineTransform()

        tform = SimilarityTransform.from_estimate(self.matched_kp_moving, self.matched_kp_ref)
        self.M = tform.params

        # Ensure homogeneous
        if self.M.shape == (2, 3):
            M33 = np.eye(3)
            M33[0:2, :] = self.M
            self.M = M33

        return self

    def warp_xy(self, xy, src_pt_level=0, dst_slide_level=0):
        """Warp points from moving image space to reference image space.

        Parameters
        ----------
        xy : ndarray
            (N, 2) points in moving image coordinates.
        src_pt_level : int
            Level of source points (unused, kept for API compatibility).
        dst_slide_level : int
            Level of destination slide (unused, kept for API compatibility).

        Returns
        -------
        warped_xy : ndarray
            (N, 2) points in reference image coordinates.
        """
        return warp_tools.warp_xy(xy, M=self.M)

    def inverse_warp_xy(self, xy):
        """Warp points from reference image space to moving image space.

        Parameters
        ----------
        xy : ndarray
            (N, 2) points in reference image coordinates.

        Returns
        -------
        warped_xy : ndarray
            (N, 2) points in moving image coordinates.
        """
        M_inv = np.linalg.inv(self.M)
        return warp_tools.warp_xy(xy, M=M_inv)

    def warp_image(self, img, out_shape_rc=None):
        """Warp an image using the estimated transform.

        Parameters
        ----------
        img : ndarray
            Image to warp.
        out_shape_rc : tuple, optional
            Output shape. Defaults to reference image shape.

        Returns
        -------
        warped_img : ndarray
            Warped image.
        """
        if out_shape_rc is None:
            out_shape_rc = self.ref_img.shape[0:2]
        return warp_tools.warp_img(img, M=self.M, out_shape_rc=out_shape_rc)


def register_pair(ref_img, moving_img, ref_name="ref", moving_name="moving",
                  feature_detector=None, matcher=None, transform_type="similarity"):
    """Convenience function to register a pair of images.

    Returns
    -------
    registrar : RigidRegistrar
        Fitted RigidRegistrar object.
    """
    registrar = RigidRegistrar(ref_img, moving_img, ref_name=ref_name, moving_name=moving_name)
    registrar.fit(feature_detector=feature_detector, matcher=matcher, transform_type=transform_type)
    return registrar

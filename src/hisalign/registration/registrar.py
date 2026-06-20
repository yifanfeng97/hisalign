"""High-level HE-to-IHC registration API.

Provides HEIHCRegistrar class that registers multiple IHC slides to a single HE reference.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from hisalign import preprocessing
from hisalign.registration import (
    feature_detectors,
    feature_matcher,
    non_rigid,
    rigid,
    warp_tools,
)
from hisalign.slide_io.base import Slide


def _read_slide_at_level(
    slide: Slide, level: int, max_dim_px: int | None = None
) -> tuple[np.ndarray, float]:
    """Read a whole slide at a given pyramid level, optionally resizing.

    Parameters
    ----------
    slide : Slide
        Slide to read.
    level : int
        Pyramid level.
    max_dim_px : int, optional
        Maximum dimension in pixels. If the slide at the given level is larger
        than this, it will be resized.

    Returns
    -------
    img : np.ndarray
        HWC uint8 RGB image.
    scale_to_level0 : float
        Scale factor converting a pixel in the returned image to level-0 pixels.
    """
    level0_w, level0_h = slide.level_dimensions[0]
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h))

    actual_w, actual_h = w, h
    if max_dim_px is not None and max(w, h) > max_dim_px:
        resize_scale = max_dim_px / max(w, h)
        actual_w, actual_h = int(w * resize_scale), int(h * resize_scale)
        img = warp_tools.resize_img(img, (actual_h, actual_w))

    scale_to_level0 = max(level0_w, level0_h) / max(actual_w, actual_h)
    return img, scale_to_level0


def _preprocess_for_registration(img: np.ndarray, method: str = "od") -> np.ndarray:
    """Preprocess a brightfield RGB image for feature-based registration.

    Parameters
    ----------
    img : np.ndarray
        HWC uint8 RGB image.
    method : str
        Either ``"od"`` for optical-density grayscale or ``"gray"`` for
        simple grayscale intensity rescaling.

    Returns
    -------
    np.ndarray
        2D uint8 grayscale image.
    """
    if method == "od":
        return preprocessing.optical_density_gray(img)

    from skimage import color as skcolor
    from skimage import exposure

    if img.ndim == 3 and img.shape[2] == 3:
        gray = skcolor.rgb2gray(img)
    else:
        gray = img
    return exposure.rescale_intensity(gray, out_range=(0, 255)).astype(np.uint8)


class HEIHCRegistrar:
    """Register multiple IHC slides to a single HE reference slide.

    Attributes
    ----------
    he_slide : Slide
        HE reference slide.
    ihc_slides : dict[str, Slide]
        Dictionary of IHC slides keyed by marker name.
    registration_level : int
        Pyramid level used for registration.
    max_image_dim_px : int
        Maximum image dimension for rigid registration.
    rigid_registrars : dict[str, rigid.RigidRegistrar]
        Rigid registrars for each IHC slide.
    non_rigid_registrars : dict[str, non_rigid.NonRigidRegistrarBase]
        Non-rigid registrars for each IHC slide.
    """

    def __init__(
        self,
        he_slide: Slide,
        ihc_slides: dict[str, Slide],
        registration_level: int = 3,
        max_image_dim_px: int = 1024,
        preprocessing_method: str = "od",
        feature_detector: feature_detectors.FeatureDD | None = None,
        matcher: feature_matcher.Matcher | None = None,
    ):
        self.he_slide = he_slide
        self.ihc_slides = ihc_slides
        self.registration_level = registration_level
        self.max_image_dim_px = max_image_dim_px

        self.rigid_registrars: dict[str, rigid.RigidRegistrar] = {}
        self.non_rigid_registrars: dict[str, non_rigid.NonRigidRegistrarBase] = {}

        self.feature_detector = feature_detector or feature_detectors.KazeFD()
        self.matcher = matcher or feature_matcher.Matcher()

        # Scale factors from processed registration image pixels to level-0 pixels
        self.he_scale_to_level0: float = 1.0
        self.ihc_scale_to_level0: dict[str, float] = {}

        # Padding matrices and common canvas shape per marker
        self.he_padding_matrix: np.ndarray | None = None
        self.ihc_padding_matrix: dict[str, np.ndarray] = {}
        self.reg_shape_rc: dict[str, tuple[int, int]] = {}

        # Read HE image at registration level
        self.he_img, self.he_scale_to_level0 = _read_slide_at_level(
            he_slide, registration_level, max_dim_px=max_image_dim_px
        )
        self.he_gray = _preprocess_for_registration(
            self.he_img, method=preprocessing_method
        )

        # Read IHC images at registration level
        self.ihc_imgs = {}
        self.ihc_grays = {}
        for marker, slide in ihc_slides.items():
            img, scale = _read_slide_at_level(
                slide, registration_level, max_dim_px=max_image_dim_px
            )
            self.ihc_imgs[marker] = img
            self.ihc_grays[marker] = _preprocess_for_registration(
                img, method=preprocessing_method
            )
            self.ihc_scale_to_level0[marker] = scale

        # Build a common registration canvas for each marker and center-pad images
        self.he_padded: np.ndarray | None = None
        self.ihc_padded: dict[str, np.ndarray] = {}
        self._build_common_canvases()

    def _build_common_canvases(self) -> None:
        """Center-pad processed HE/IHC images to a common canvas per marker."""
        for marker, ihc_gray in self.ihc_grays.items():
            reg_h = max(self.he_gray.shape[0], ihc_gray.shape[0])
            reg_w = max(self.he_gray.shape[1], ihc_gray.shape[1])
            reg_shape_rc = (reg_h, reg_w)
            self.reg_shape_rc[marker] = reg_shape_rc

            he_tform = warp_tools.get_padding_matrix(self.he_gray.shape, reg_shape_rc)
            ihc_tform = warp_tools.get_padding_matrix(ihc_gray.shape, reg_shape_rc)
            self.he_padding_matrix = he_tform
            self.ihc_padding_matrix[marker] = ihc_tform

            he_padded = warp_tools.warp_img(
                self.he_gray, M=he_tform, out_shape_rc=reg_shape_rc
            )
            ihc_padded = warp_tools.warp_img(
                ihc_gray, M=ihc_tform, out_shape_rc=reg_shape_rc
            )

            if self.he_padded is None:
                self.he_padded = he_padded
            self.ihc_padded[marker] = ihc_padded

    def fit(self) -> "HEIHCRegistrar":
        """Run rigid + non-rigid registration for all IHC slides.

        Returns
        -------
        self
        """
        for marker, ihc_padded in tqdm(
            self.ihc_padded.items(), desc="Registering IHC slides"
        ):
            # Rigid registration on the common padded canvas
            rigid_reg = rigid.RigidRegistrar(
                ref_img=self.he_padded,
                moving_img=ihc_padded,
                ref_name="HE",
                moving_name=marker,
            )
            rigid_reg.fit(
                feature_detector=self.feature_detector,
                matcher=self.matcher,
                transform_type="similarity",
            )
            self.rigid_registrars[marker] = rigid_reg

            # Non-rigid registration on the same padded canvas
            nr_reg = non_rigid.NonRigidRegistrar(
                ref_img=self.he_padded,
                moving_img=ihc_padded,
                M=rigid_reg.M,
                ref_name="HE",
                moving_name=marker,
            )
            nr_reg.fit(non_rigid_reg_class=non_rigid.OpticalFlowWarper)
            self.non_rigid_registrars[marker] = nr_reg

        return self

    def _to_padded_he(self, xy: np.ndarray) -> np.ndarray:
        """Map points from unpadded HE processed coordinates to padded common canvas."""
        if self.he_padding_matrix is None:
            return xy
        return warp_tools.warp_xy(xy, M=self.he_padding_matrix)

    def _from_padded_he(self, xy: np.ndarray) -> np.ndarray:
        """Map points from padded common canvas to unpadded HE processed coordinates."""
        if self.he_padding_matrix is None:
            return xy
        return warp_tools.warp_xy(xy, M=np.linalg.inv(self.he_padding_matrix))

    def _to_padded_ihc(self, xy: np.ndarray, marker: str) -> np.ndarray:
        """Map points from unpadded IHC processed coordinates to padded common canvas."""
        ihc_tform = self.ihc_padding_matrix.get(marker)
        if ihc_tform is None:
            return xy
        return warp_tools.warp_xy(xy, M=ihc_tform)

    def _from_padded_ihc(self, xy: np.ndarray, marker: str) -> np.ndarray:
        """Map points from padded common canvas to unpadded IHC processed coordinates."""
        ihc_tform = self.ihc_padding_matrix.get(marker)
        if ihc_tform is None:
            return xy
        return warp_tools.warp_xy(xy, M=np.linalg.inv(ihc_tform))

    def warp_xy_from_he_to_ihc(
        self,
        xy: np.ndarray,
        marker: str,
        src_pt_level: int = 0,
        dst_slide_level: int = 0,
    ) -> np.ndarray:
        """Map points from HE level-0 to IHC level-0.

        The non-rigid registrar operates on the padded common canvas, so the
        mapping first pads HE coordinates, applies the inverse registration
        transform, then un-pads the IHC coordinates.
        """
        if marker not in self.non_rigid_registrars:
            raise KeyError(
                f"Marker '{marker}' not found. Available: {list(self.non_rigid_registrars.keys())}"
            )

        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)

        nr_reg = self.non_rigid_registrars[marker]

        he_scale = self.he_scale_to_level0
        ihc_scale = self.ihc_scale_to_level0[marker]

        xy_reg = xy / he_scale
        xy_padded = self._to_padded_he(xy_reg)
        ihc_padded = nr_reg.inverse_warp_xy(xy_padded)
        ihc_reg = self._from_padded_ihc(ihc_padded, marker)
        ihc_xy = ihc_reg * ihc_scale

        return ihc_xy

    def warp_xy_from_ihc_to_he(
        self,
        xy: np.ndarray,
        marker: str,
        src_pt_level: int = 0,
        dst_slide_level: int = 0,
    ) -> np.ndarray:
        """Map points from IHC level-0 to HE level-0."""
        if marker not in self.non_rigid_registrars:
            raise KeyError(
                f"Marker '{marker}' not found. Available: {list(self.non_rigid_registrars.keys())}"
            )

        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)

        nr_reg = self.non_rigid_registrars[marker]

        he_scale = self.he_scale_to_level0
        ihc_scale = self.ihc_scale_to_level0[marker]

        xy_reg = xy / ihc_scale
        xy_padded = self._to_padded_ihc(xy_reg, marker)
        he_padded = nr_reg.warp_xy(xy_padded)
        he_reg = self._from_padded_he(he_padded)
        he_xy = he_reg * he_scale

        return he_xy

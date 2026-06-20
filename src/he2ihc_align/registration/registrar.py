"""High-level HE-to-IHC registration API.

Provides HEIHCRegistrar class that registers multiple IHC slides to a single HE reference.
"""

from __future__ import annotations

import numpy as np
from skimage import color as skcolor
from skimage import exposure
from tqdm import tqdm

from he2ihc_align.registration import (
    non_rigid,
    rigid,
    warp_tools,
)
from he2ihc_align.slide_io.base import Slide


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


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale uint8."""
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
    max_non_rigid_dim_px : int
        Maximum image dimension for non-rigid registration.
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
        max_non_rigid_dim_px: int = 2048,
    ):
        self.he_slide = he_slide
        self.ihc_slides = ihc_slides
        self.registration_level = registration_level
        self.max_image_dim_px = max_image_dim_px
        self.max_non_rigid_dim_px = max_non_rigid_dim_px

        self.rigid_registrars: dict[str, rigid.RigidRegistrar] = {}
        self.non_rigid_registrars: dict[str, non_rigid.NonRigidRegistrarBase] = {}

        # Scale factors from processed registration image pixels to level-0 pixels
        self.he_scale_to_level0: float = 1.0
        self.ihc_scale_to_level0: dict[str, float] = {}

        # Read HE image at registration level
        self.he_img, self.he_scale_to_level0 = _read_slide_at_level(
            he_slide, registration_level, max_dim_px=max_image_dim_px
        )
        self.he_gray = _to_grayscale(self.he_img)

        # Read IHC images at registration level
        self.ihc_imgs = {}
        self.ihc_grays = {}
        for marker, slide in ihc_slides.items():
            img, scale = _read_slide_at_level(slide, registration_level, max_dim_px=max_image_dim_px)
            self.ihc_imgs[marker] = img
            self.ihc_grays[marker] = _to_grayscale(img)
            self.ihc_scale_to_level0[marker] = scale

    def fit(self) -> "HEIHCRegistrar":
        """Run rigid + non-rigid registration for all IHC slides.

        Returns
        -------
        self
        """
        for marker, ihc_gray in tqdm(self.ihc_grays.items(), desc="Registering IHC slides"):
            # Rigid registration
            rigid_reg = rigid.RigidRegistrar(
                ref_img=self.he_gray,
                moving_img=ihc_gray,
                ref_name="HE",
                moving_name=marker,
            )
            rigid_reg.fit(transform_type="similarity")
            self.rigid_registrars[marker] = rigid_reg

            # Non-rigid registration
            # Use higher resolution for non-rigid if requested
            if self.max_non_rigid_dim_px > self.max_image_dim_px:
                he_nr, _ = _read_slide_at_level(
                    self.he_slide, self.registration_level, max_dim_px=self.max_non_rigid_dim_px
                )
                he_nr_gray = _to_grayscale(he_nr)
                ihc_nr, _ = _read_slide_at_level(
                    self.ihc_slides[marker], self.registration_level, max_dim_px=self.max_non_rigid_dim_px
                )
                ihc_nr_gray = _to_grayscale(ihc_nr)

                # Scale rigid transform to new resolution.
                # The transform maps moving image coordinates to reference image
                # coordinates.  When both images are upsampled, the linear part
                # only needs to account for the difference in upsampling ratios,
                # while the translation (in reference pixels) scales with the
                # reference upsampling factor.
                ref_scale_factor = max(he_nr_gray.shape) / max(self.he_gray.shape)
                moving_scale_factor = max(ihc_nr_gray.shape) / max(self.ihc_grays[marker].shape)
                m_scaled = rigid_reg.M.copy()
                scale_ratio = ref_scale_factor / moving_scale_factor
                m_scaled[0, 0] *= scale_ratio
                m_scaled[0, 1] *= scale_ratio
                m_scaled[1, 0] *= scale_ratio
                m_scaled[1, 1] *= scale_ratio
                m_scaled[0, 2] *= ref_scale_factor
                m_scaled[1, 2] *= ref_scale_factor
            else:
                he_nr_gray = self.he_gray
                ihc_nr_gray = ihc_gray
                m_scaled = rigid_reg.M.copy()

            nr_reg = non_rigid.NonRigidRegistrar(
                ref_img=he_nr_gray,
                moving_img=ihc_nr_gray,
                M=m_scaled,
                ref_name="HE",
                moving_name=marker,
            )
            nr_reg.fit(non_rigid_reg_class=non_rigid.OpticalFlowWarper)
            self.non_rigid_registrars[marker] = nr_reg

        return self

    def warp_xy_from_he_to_ihc(
        self,
        xy: np.ndarray,
        marker: str,
        src_pt_level: int = 0,
        dst_slide_level: int = 0,
    ) -> np.ndarray:
        """Map points from HE level-0 to IHC level-0.

        This applies the inverse of the registration transform:
        1. Scale from HE level-0 to registration level
        2. Apply inverse non-rigid displacement
        3. Apply inverse rigid transform
        4. Scale from registration level to IHC level-0

        Parameters
        ----------
        xy : np.ndarray
            (N, 2) points in HE level-0 coordinates.
        marker : str
            Marker name of the IHC slide.
        src_pt_level : int
            Level of source points (HE). Default 0.
        dst_slide_level : int
            Level of destination slide (IHC). Default 0.

        Returns
        -------
        ihc_xy : np.ndarray
            (N, 2) points in IHC level-0 coordinates.
        """
        if marker not in self.non_rigid_registrars:
            raise KeyError(f"Marker '{marker}' not found. Available: {list(self.non_rigid_registrars.keys())}")

        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)

        nr_reg = self.non_rigid_registrars[marker]

        # Get scale factors between level 0 and registration level
        he_downsample = self.he_slide.level_downsamples[self.registration_level]
        ihc_downsample = self.ihc_slides[marker].level_downsamples[self.registration_level]

        # Scale from HE level-0 to registration level
        xy_reg = xy / he_downsample

        # Apply inverse non-rigid + rigid (reference to moving)
        ihc_xy_reg = nr_reg.inverse_warp_xy(xy_reg)

        # Scale from registration level to IHC level-0
        ihc_xy = ihc_xy_reg * ihc_downsample

        return ihc_xy

    def warp_xy_from_ihc_to_he(
        self,
        xy: np.ndarray,
        marker: str,
        src_pt_level: int = 0,
        dst_slide_level: int = 0,
    ) -> np.ndarray:
        """Map points from IHC level-0 to HE level-0.

        This applies the forward registration transform.

        Parameters
        ----------
        xy : np.ndarray
            (N, 2) points in IHC level-0 coordinates.
        marker : str
            Marker name of the IHC slide.
        src_pt_level : int
            Level of source points (IHC). Default 0.
        dst_slide_level : int
            Level of destination slide (HE). Default 0.

        Returns
        -------
        he_xy : np.ndarray
            (N, 2) points in HE level-0 coordinates.
        """
        if marker not in self.non_rigid_registrars:
            raise KeyError(f"Marker '{marker}' not found. Available: {list(self.non_rigid_registrars.keys())}")

        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)

        nr_reg = self.non_rigid_registrars[marker]

        # Get scale factors
        he_downsample = self.he_slide.level_downsamples[self.registration_level]
        ihc_downsample = self.ihc_slides[marker].level_downsamples[self.registration_level]

        # Scale from IHC level-0 to registration level
        xy_reg = xy / ihc_downsample

        # Apply rigid + non-rigid (moving to reference)
        he_xy_reg = nr_reg.warp_xy(xy_reg)

        # Scale from registration level to HE level-0
        he_xy = he_xy_reg * he_downsample

        return he_xy

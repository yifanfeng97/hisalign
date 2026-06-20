"""Simplified non-rigid registration for HE-to-IHC alignment.

Based on VALIS non_rigid_registrars.py and serial_non_rigid.py, trimmed to MVP.
Uses optical flow for non-rigid registration.
Simplified to single reference + N moving images.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import warp_tools


class NonRigidRegistrarBase:
    """Abstract base class for non-rigid registration using displacement fields.

    Warps moving_img to align with fixed_img using backwards transformations.
    """

    def __init__(self, params=None):
        self.params = params or {}
        self.moving_img = None
        self.fixed_img = None
        self.mask = None
        self.shape = None
        self.warped_image = None
        self.backward_dx = None
        self.backward_dy = None
        self.method = None

    def calc(self, moving_img, fixed_img, mask=None, **kwargs):
        """Calculate displacement fields.

        Returns
        -------
        bk_dxdy : list
            [dx, dy] backwards displacement field.
        """
        raise NotImplementedError

    def register(self, moving_img, fixed_img, mask=None, **kwargs):
        """Register images, warping moving_img to align with fixed_img.

        Returns
        -------
        warped_img : ndarray
            Warped moving image.
        warp_grid : ndarray
            Image showing deformation applied to regular grid.
        bk_dxdy : list
            [dx, dy] backwards displacement field.
        """
        moving_shape = warp_tools.get_shape(moving_img)[0:2]
        fixed_shape = warp_tools.get_shape(fixed_img)[0:2]
        assert np.all(moving_shape == fixed_shape), "Images have different shapes"

        self.shape = moving_shape
        self.moving_img = moving_img
        self.fixed_img = fixed_img

        if mask is None:
            mask = np.full(self.shape, 255, dtype=np.uint8)
        self.mask = mask

        # Apply mask
        if self.mask is not None:
            mask_bbox = warp_tools.xy2bbox(warp_tools.mask2xy(self.mask))
            min_c, min_r = mask_bbox[0:2]
            max_c, max_r = mask_bbox[0:2] + mask_bbox[2:]
            mask = self.mask[min_r:max_r, min_c:max_c]
            masked_moving = moving_img[min_r:max_r, min_c:max_c].copy()
            masked_fixed = fixed_img[min_r:max_r, min_c:max_c].copy()
        else:
            masked_moving = moving_img.copy()
            masked_fixed = fixed_img.copy()

        bk_dxdy = self.calc(
            moving_img=masked_moving, fixed_img=masked_fixed, mask=mask, **kwargs
        )

        if self.mask is not None and bk_dxdy is not None:
            bk_dx = np.zeros(self.shape)
            bk_dx[min_r:max_r, min_c:max_c] = bk_dxdy[0]
            bk_dx[self.mask == 0] = 0

            bk_dy = np.zeros(self.shape)
            bk_dy[min_r:max_r, min_c:max_c] = bk_dxdy[1]
            bk_dy[self.mask == 0] = 0

            bk_dxdy = [bk_dx, bk_dy]

        if bk_dxdy is not None:
            warped_img = warp_tools.warp_img(moving_img, bk_dxdy=bk_dxdy)
            grid_img = self.get_grid_image()
            warp_grid = warp_tools.warp_img(grid_img, bk_dxdy=bk_dxdy)
        else:
            warped_img = moving_img.copy()
            warp_grid = self.get_grid_image()

        self.backward_dx = bk_dxdy[0] if bk_dxdy is not None else None
        self.backward_dy = bk_dxdy[1] if bk_dxdy is not None else None
        self.warped_image = warped_img

        return warped_img, warp_grid, bk_dxdy

    def get_grid_image(self, grid_spacing=16, thickness=1):
        """Create an image of a regular grid."""
        grid_img = np.zeros(self.shape[:2])
        for r in range(0, self.shape[0], grid_spacing):
            grid_img[r : r + thickness, :] = 255
        for c in range(0, self.shape[1], grid_spacing):
            grid_img[:, c : c + thickness] = 255
        return grid_img


class OpticalFlowWarper(NonRigidRegistrarBase):
    """Use dense optical flow to register images.

    Uses OpenCV's DeepFlow or Farneback optical flow.
    """

    def __init__(
        self, optical_flow_obj=None, smoothing_method="gauss", sigma_ratio=0.005
    ):
        """
        Parameters
        ----------
        optical_flow_obj : object
            Optical flow object. If None, uses cv2.optflow.createOptFlow_DeepFlow() if available,
            otherwise cv2.calcOpticalFlowFarneback.
        smoothing_method : str
            "gauss" for Gaussian smoothing, "None" for no smoothing.
        sigma_ratio : float
            Sigma ratio for Gaussian smoothing.
        """
        super().__init__()
        self.smoothing_method = smoothing_method
        self.sigma_ratio = sigma_ratio

        if optical_flow_obj is None:
            try:
                self.optical_flow_obj = cv2.optflow.createOptFlow_DeepFlow()
                self.method = "DeepFlow"
            except (AttributeError, cv2.error):
                self.optical_flow_obj = None
                self.method = "Farneback"
        else:
            self.optical_flow_obj = optical_flow_obj
            self.method = optical_flow_obj.__class__.__name__

    def calc(self, moving_img, fixed_img, mask=None, **kwargs):
        """Calculate optical flow displacement field.

        Returns
        -------
        bk_dxdy : list
            [dx, dy] backwards displacement field.
        """
        # Convert to grayscale if needed
        if moving_img.ndim == 3:
            moving_gray = cv2.cvtColor(moving_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            moving_gray = moving_img.astype(np.uint8)

        if fixed_img.ndim == 3:
            fixed_gray = cv2.cvtColor(fixed_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            fixed_gray = fixed_img.astype(np.uint8)

        # Calculate optical flow
        if self.optical_flow_obj is not None:
            flow = self.optical_flow_obj.calc(moving_gray, fixed_gray, None)
        else:
            # Use Farneback as fallback
            flow = cv2.calcOpticalFlowFarneback(
                moving_gray,
                fixed_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

        # flow is (H, W, 2) where flow[..., 0] is dx and flow[..., 1] is dy
        dx = flow[..., 0]
        dy = flow[..., 1]

        # Smooth if requested
        if self.smoothing_method == "gauss" and self.sigma_ratio > 0:
            bk_dxdy = warp_tools.smooth_dxdy([dx, dy], sigma_ratio=self.sigma_ratio)
        else:
            bk_dxdy = [dx, dy]

        return bk_dxdy


class NonRigidRegistrarXY(NonRigidRegistrarBase):
    """Non-rigid registration that can use corresponding points."""

    def __init__(self, params=None):
        super().__init__(params=params)
        self.moving_xy = None
        self.fixed_xy = None

    def register(
        self, moving_img, fixed_img, mask=None, moving_xy=None, fixed_xy=None, **kwargs
    ):
        """Register with optional corresponding points."""
        if moving_xy is not None and fixed_xy is not None:
            moving_xy, fixed_xy = self.filter_xy(
                moving_xy, fixed_xy, moving_img.shape, mask
            )

        self.moving_xy = moving_xy
        self.fixed_xy = fixed_xy
        return NonRigidRegistrarBase.register(
            self, moving_img=moving_img, fixed_img=fixed_img, mask=mask, **kwargs
        )

    def filter_xy(self, moving_xy, fixed_xy, img_shape_rc, mask=None):
        """Remove points outside image and/or mask."""
        if mask is None:
            mask = np.full(img_shape_rc, 255, dtype=np.uint8)
        moving_inside_idx = warp_tools.get_inside_mask_idx(moving_xy, mask)
        fixed_inside_idx = warp_tools.get_inside_mask_idx(fixed_xy, mask)
        inside_idx = np.intersect1d(moving_inside_idx, fixed_inside_idx)
        return moving_xy[inside_idx, :], fixed_xy[inside_idx, :]


class SimpleElastixWarper(NonRigidRegistrarXY):
    """Uses SimpleElastix/SimpleITK to register images.

    Requires SimpleITK to be installed.
    """

    def __init__(
        self, params=None, ammi_weight=0.33, bending_penalty_weight=0.33, kp_weight=0.33
    ):
        super().__init__(params=params)
        self.ammi_weight = ammi_weight
        self.bending_penalty_weight = bending_penalty_weight
        self.kp_weight = kp_weight
        if params is not None:
            self._params_provided = True
            self.params = params
        else:
            self._params_provided = False

    @staticmethod
    def get_default_params(img_shape, grid_spacing_ratio=0.025):
        """Get default Elastix parameters."""
        import SimpleITK as sitk  # noqa: N813

        p = sitk.GetDefaultParameterMap("bspline")
        p["Metric"] = [
            "AdvancedMattesMutualInformation",
            "TransformBendingEnergyPenalty",
        ]
        p["MaximumNumberOfIterations"] = ["1500"]
        p["FixedImagePyramid"] = ["FixedRecursiveImagePyramid"]
        p["MovingImagePyramid"] = ["MovingRecursiveImagePyramid"]
        p["Interpolator"] = ["BSplineInterpolator"]
        p["ImageSampler"] = ["RandomCoordinate"]
        p["MetricSamplingStrategy"] = ["None"]
        p["UseRandomSampleRegion"] = ["true"]
        p["ErodeMask"] = ["true"]
        p["NumberOfHistogramBins"] = ["32"]
        p["NumberOfSpatialSamples"] = ["3000"]
        p["NewSamplesEveryIteration"] = ["true"]
        p["SampleRegionSize"] = [str(min([img_shape[1] // 3, img_shape[0] // 3]))]
        p["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
        p["ASGDParameterEstimationMethod"] = ["DisplacementDistribution"]
        p["HowToCombineTransforms"] = ["Compose"]
        grid_spacing = str(
            int(
                np.mean(
                    [
                        img_shape[1] * grid_spacing_ratio,
                        img_shape[0] * grid_spacing_ratio,
                    ]
                )
            )
        )
        p["FinalGridSpacingInPhysicalUnits"] = [grid_spacing]
        p["WriteResultImage"] = ["false"]
        return p

    def calc(
        self, moving_img, fixed_img, mask=None, moving_xy=None, fixed_xy=None, **kwargs
    ):
        """Calculate non-rigid registration using SimpleElastix."""
        import SimpleITK as sitk  # noqa: N813  # noqa: N813

        assert moving_img.shape == fixed_img.shape, "Images have different shapes"

        if not self._params_provided:
            self.params = self.get_default_params(moving_img.shape)

        # Run Elastix
        elastix_image_filter = sitk.ElastixImageFilter()
        sitk_moving = sitk.GetImageFromArray(moving_img)
        sitk_fixed = sitk.GetImageFromArray(fixed_img)
        elastix_image_filter.SetMovingImage(sitk_moving)
        elastix_image_filter.SetFixedImage(sitk_fixed)
        elastix_image_filter.SetParameterMap(self.params)

        if mask is not None:
            sitk_mask = sitk.Cast(
                sitk.GetImageFromArray(mask.astype(np.uint8)), sitk.sitkUInt8
            )
            elastix_image_filter.SetFixedMask(sitk_mask)

        elastix_image_filter.Execute()

        # Get deformation field
        transformix = sitk.TransformixImageFilter()
        transformix.SetTransformParameterMap(
            elastix_image_filter.GetTransformParameterMap()
        )
        transformix.ComputeDeformationFieldOn()
        transformix.Execute()
        deformation = sitk.GetArrayFromImage(transformix.GetDeformationField())

        dxdy = np.array([deformation[..., 0], deformation[..., 1]])
        return dxdy


class NonRigidRegistrar:
    """Simplified non-rigid registrar for single reference + moving pair.

    Attributes
    ----------
    ref_img : ndarray
        Reference image.
    moving_img : ndarray
        Moving image.
    bk_dxdy : list
        Backwards displacement field [dx, dy].
    fwd_dxdy : list
        Forward displacement field [dx, dy].
    M : ndarray
        Rigid transformation matrix.
    """

    def __init__(
        self, ref_img, moving_img, M=None, ref_name="ref", moving_name="moving"
    ):  # noqa: N803
        self.ref_img = ref_img
        self.moving_img = moving_img
        self.ref_name = ref_name
        self.moving_name = moving_name
        self.M = M if M is not None else np.eye(3)
        self.bk_dxdy = None
        self.fwd_dxdy = None
        self.warped_img = None
        self.non_rigid_reg_obj = None

    def fit(self, non_rigid_reg_class=OpticalFlowWarper, non_rigid_reg_params=None):
        """Run non-rigid registration.

        Parameters
        ----------
        non_rigid_reg_class : type
            Non-rigid registrar class. Defaults to OpticalFlowWarper.
        non_rigid_reg_params : dict, optional
            Parameters for non-rigid registrar.

        Returns
        -------
        self
        """
        if non_rigid_reg_params is None:
            non_rigid_reg_params = {}

        # Warp moving image with rigid transform first
        rigid_warped = warp_tools.warp_img(
            self.moving_img, M=self.M, out_shape_rc=self.ref_img.shape[0:2]
        )

        # Create non-rigid registrar
        if isinstance(non_rigid_reg_class, type):
            self.non_rigid_reg_obj = non_rigid_reg_class(**non_rigid_reg_params)
        else:
            self.non_rigid_reg_obj = non_rigid_reg_class

        # Register
        warped_img, warp_grid, bk_dxdy = self.non_rigid_reg_obj.register(
            moving_img=rigid_warped,
            fixed_img=self.ref_img,
        )

        self.warped_img = warped_img
        if bk_dxdy is not None:
            self.bk_dxdy = bk_dxdy
            self.fwd_dxdy = warp_tools.get_inverse_field(bk_dxdy)

        return self

    def warp_xy(self, xy, src_pt_level=0, dst_slide_level=0):
        """Warp points from moving to reference space (including non-rigid).

        First applies rigid transform, then backwards displacement.
        """
        # Apply rigid
        rigid_xy = warp_tools.warp_xy(xy, M=self.M)
        # Apply non-rigid backwards displacement
        if self.bk_dxdy is not None:
            return warp_tools.warp_xy(rigid_xy, bk_dxdy=self.bk_dxdy)
        return rigid_xy

    def inverse_warp_xy(self, xy):
        """Warp points from reference to moving space (including non-rigid).

        First applies forward displacement, then inverse rigid.
        """
        # Apply forward non-rigid displacement
        if self.fwd_dxdy is not None:
            nr_xy = warp_tools.warp_xy(xy, fwd_dxdy=self.fwd_dxdy)
        else:
            nr_xy = xy.copy()

        # Apply inverse rigid
        m_inv = np.linalg.inv(self.M)
        return warp_tools.warp_xy(nr_xy, M=m_inv)

    def warp_image(self, img, out_shape_rc=None):
        """Warp image using both rigid and non-rigid transforms."""
        if out_shape_rc is None:
            out_shape_rc = self.ref_img.shape[0:2]
        # Apply rigid then non-rigid
        rigid_warped = warp_tools.warp_img(img, M=self.M, out_shape_rc=out_shape_rc)
        if self.bk_dxdy is not None:
            return warp_tools.warp_img(rigid_warped, bk_dxdy=self.bk_dxdy)
        return rigid_warped

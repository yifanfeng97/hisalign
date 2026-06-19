"""Simplified warp tools for HE-to-IHC registration.

Based on VALIS warp_tools.py, trimmed to MVP.
Keeps only essential warping, transformation, and utility functions.
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage
from skimage import transform


def get_shape(img):
    """Get shape of image (row, col)."""
    if img is None:
        return None
    return img.shape[0:2]


def rescale_img(img, s):
    """Rescale image by factor s."""
    if img is None:
        return None
    new_shape = (int(img.shape[1] * s), int(img.shape[0] * s))
    return cv2.resize(img, new_shape)


def resize_img(img, new_shape_rc, interp_method="linear"):
    """Resize image to new shape (row, col)."""
    if img is None:
        return None
    if isinstance(new_shape_rc, (list, tuple, np.ndarray)):
        new_shape = (int(new_shape_rc[1]), int(new_shape_rc[0]))
    else:
        new_shape = (int(new_shape_rc), int(new_shape_rc))
    if interp_method == "nearest":
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_LINEAR
    return cv2.resize(img, new_shape, interpolation=interp)


def get_padding_matrix(src_shape_rc, dst_shape_rc):
    """Get transformation matrix to pad image to dst_shape."""
    tx = (dst_shape_rc[1] - src_shape_rc[1]) / 2
    ty = (dst_shape_rc[0] - src_shape_rc[0]) / 2
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
    ], dtype=np.float64)
    return T


def get_corners_of_image(shape_rc):
    """Get corners of image in (row, col) order."""
    r, c = shape_rc[0], shape_rc[1]
    corners = np.array([
        [0, 0],
        [0, c],
        [r, c],
        [r, 0],
    ], dtype=np.float64)
    return corners


def warp_xy(xy, M=None, bk_dxdy=None, fwd_dxdy=None):
    """Warp points using affine matrix and/or displacement field.

    Parameters
    ----------
    xy : ndarray
        (N, 2) array of points in xy coordinates.
    M : ndarray, optional
        3x3 affine transformation matrix.
    bk_dxdy : list, optional
        [dx, dy] backwards displacement field.
    fwd_dxdy : list, optional
        [dx, dy] forward displacement field.

    Returns
    -------
    warped_xy : ndarray
        (N, 2) array of warped points.
    """
    if xy is None or len(xy) == 0:
        return xy

    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim == 1:
        xy = xy.reshape(1, -1)

    warped = xy.copy()

    # Apply affine transform
    if M is not None:
        ones = np.ones((warped.shape[0], 1))
        homog = np.hstack([warped, ones])
        warped = (M @ homog.T).T[:, 0:2]

    # Apply forward displacement (for warping from moving to fixed)
    if fwd_dxdy is not None:
        dx, dy = fwd_dxdy[0], fwd_dxdy[1]
        # Sample displacement at point locations
        x_coords = np.clip(warped[:, 0].astype(int), 0, dx.shape[1] - 1)
        y_coords = np.clip(warped[:, 1].astype(int), 0, dx.shape[0] - 1)
        warped[:, 0] += dx[y_coords, x_coords]
        warped[:, 1] += dy[y_coords, x_coords]

    # Apply backwards displacement (for warping from fixed to moving)
    if bk_dxdy is not None:
        dx, dy = bk_dxdy[0], bk_dxdy[1]
        x_coords = np.clip(warped[:, 0].astype(int), 0, dx.shape[1] - 1)
        y_coords = np.clip(warped[:, 1].astype(int), 0, dx.shape[0] - 1)
        warped[:, 0] -= dx[y_coords, x_coords]
        warped[:, 1] -= dy[y_coords, x_coords]

    return warped


def warp_xy_from_to(xy, src_M, src_bk_dxdy, dst_M, dst_bk_dxdy,
                    src_fwd_dxdy=None, dst_fwd_dxdy=None):
    """Warp points from src image space to dst image space.

    First unwarp from src (apply inverse of src transforms), then warp to dst.
    """
    if xy is None or len(xy) == 0:
        return xy

    # For simplicity, we warp src to a common reference, then to dst
    # Step 1: Warp src points to reference using src transforms
    ref_xy = warp_xy(xy, M=src_M, fwd_dxdy=src_fwd_dxdy)

    # Step 2: Unwarp from reference to dst (apply inverse of dst transforms)
    # We need to invert dst_M
    if dst_M is not None:
        dst_M_inv = np.linalg.inv(dst_M)
    else:
        dst_M_inv = None

    # To go from reference to dst, we apply the inverse of dst's forward warping
    # dst warps its points to ref using (M, fwd_dxdy)
    # So to go from ref to dst, we apply inverse
    dst_xy = warp_xy(ref_xy, M=dst_M_inv, bk_dxdy=dst_bk_dxdy)

    return dst_xy


def warp_img(img, M=None, bk_dxdy=None, out_shape_rc=None, interp_method="linear", bg_color=None):
    """Warp image using affine matrix and/or backwards displacement field.

    Parameters
    ----------
    img : ndarray
        Image to warp.
    M : ndarray, optional
        3x3 affine transformation matrix.
    bk_dxdy : list, optional
        [dx, dy] backwards displacement field.
    out_shape_rc : tuple, optional
        Output shape (row, col).
    interp_method : str
        Interpolation method.
    bg_color : int or tuple, optional
        Background color.

    Returns
    -------
    warped_img : ndarray
        Warped image.
    """
    if img is None:
        return None

    if out_shape_rc is None:
        out_shape_rc = img.shape[0:2]

    # Apply affine transform first
    if M is not None:
        if interp_method == "nearest":
            flags = cv2.INTER_NEAREST
        else:
            flags = cv2.INTER_LINEAR
        dsize = (int(out_shape_rc[1]), int(out_shape_rc[0]))
        warped = cv2.warpAffine(img, M[0:2, :], dsize=dsize, flags=flags,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color or 0)
    else:
        warped = img.copy()
        if warped.shape[0:2] != tuple(out_shape_rc):
            warped = resize_img(warped, out_shape_rc, interp_method=interp_method)

    # Apply backwards displacement
    if bk_dxdy is not None:
        dx, dy = bk_dxdy[0], bk_dxdy[1]
        # Create remap coordinates
        h, w = warped.shape[0:2]
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x - dx).astype(np.float32)
        map_y = (map_y - dy).astype(np.float32)

        if interp_method == "nearest":
            interp = cv2.INTER_NEAREST
        else:
            interp = cv2.INTER_LINEAR

        if warped.ndim == 3:
            warped = cv2.remap(warped, map_x, map_y, interpolation=interp,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color or 0)
        else:
            warped = cv2.remap(warped, map_x, map_y, interpolation=interp,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color or 0)

    return warped


def get_inverse_field(bk_dxdy, n_inter=10):
    """Invert displacement field using iterative method.

    Parameters
    ----------
    bk_dxdy : list
        [dx, dy] backwards displacement field.
    n_inter : int
        Number of iterations.

    Returns
    -------
    fwd_dxdy : list
        [dx, dy] forward displacement field.
    """
    dx, dy = bk_dxdy[0], bk_dxdy[1]
    h, w = dx.shape

    # Initialize forward displacement as negative of backward
    fwd_dx = -dx.copy()
    fwd_dy = -dy.copy()

    # Iteratively refine
    for _ in range(n_inter):
        # Sample backward displacement at forward-warped locations
        x_coords = np.clip(np.arange(w) + fwd_dx, 0, w - 1).astype(np.float32)
        y_coords = np.clip(np.arange(h)[:, None] + fwd_dy, 0, h - 1).astype(np.float32)

        # Use ndimage.map_coordinates for sampling
        sampled_dx = ndimage.map_coordinates(dx, [y_coords, x_coords], order=1, mode='constant')
        sampled_dy = ndimage.map_coordinates(dy, [y_coords, x_coords], order=1, mode='constant')

        # Update forward displacement
        fwd_dx = -sampled_dx
        fwd_dy = -sampled_dy

    return [fwd_dx, fwd_dy]


def remove_invasive_displacements(bk_dxdy, M, src_shape_rc, out_shape_rc, inpaint_holes=False):
    """Remove displacements that would distort image edges.

    Simplified version that zeros out displacements outside the affine mask.
    """
    new_dx = bk_dxdy[0].copy()
    new_dy = bk_dxdy[1].copy()

    if M is not None:
        # Create mask of where the affine-transformed image would be
        affine_mask = warp_img(np.full(src_shape_rc, 255, dtype=np.uint8), M,
                               out_shape_rc=out_shape_rc, interp_method="nearest")
        if not np.all(out_shape_rc == bk_dxdy[0].shape):
            affine_mask = resize_img(affine_mask, bk_dxdy[0].shape, interp_method="nearest")
        new_dx[affine_mask == 0] = 0
        new_dy[affine_mask == 0] = 0
    else:
        affine_mask = np.full(out_shape_rc, 255, dtype=np.uint8)

    return [new_dx, new_dy]


def calc_d(xy1, xy2):
    """Calculate Euclidean distance between corresponding points."""
    return np.sqrt(np.sum((xy1 - xy2) ** 2, axis=1))


def measure_error(moving_xy, fixed_xy, shape_rc):
    """Measure registration error between corresponding points.

    Returns TRE (target registration error) and median distance.
    """
    if moving_xy is None or fixed_xy is None or len(moving_xy) == 0:
        return 0.0, 0.0
    d = calc_d(moving_xy, fixed_xy)
    tre = np.mean(d) / np.sqrt(shape_rc[0] * shape_rc[1]) if len(d) > 0 else 0.0
    med_d = np.median(d) if len(d) > 0 else 0.0
    return tre, med_d


def calc_total_error(error_list):
    """Calculate total error from list of errors."""
    if error_list is None or len(error_list) == 0:
        return 0.0
    valid_errors = [e for e in error_list if e is not None and not np.isnan(e)]
    if len(valid_errors) == 0:
        return 0.0
    return np.sum(valid_errors)


def mask2xy(mask):
    """Get xy coordinates of non-zero pixels in mask."""
    if mask is None:
        return np.zeros((0, 2))
    y, x = np.where(mask > 0)
    return np.column_stack([x, y])


def xy2bbox(xy):
    """Get bounding box from xy coordinates: (min_x, min_y, width, height)."""
    if len(xy) == 0:
        return np.array([0, 0, 0, 0])
    min_x, min_y = np.min(xy, axis=0)
    max_x, max_y = np.max(xy, axis=0)
    return np.array([min_x, min_y, max_x - min_x, max_y - min_y])


def bbox2mask(min_x, min_y, w, h, shape_rc):
    """Create mask from bounding box."""
    mask = np.zeros(shape_rc, dtype=np.uint8)
    min_r = max(0, int(min_y))
    max_r = min(shape_rc[0], int(min_y + h))
    min_c = max(0, int(min_x))
    max_c = min(shape_rc[1], int(min_x + w))
    mask[min_r:max_r, min_c:max_c] = 255
    return mask


def get_inside_mask_idx(xy, mask):
    """Get indices of points inside mask."""
    if mask is None or len(xy) == 0:
        return np.arange(len(xy))
    xy = np.asarray(xy)
    x = np.clip(xy[:, 0].astype(int), 0, mask.shape[1] - 1)
    y = np.clip(xy[:, 1].astype(int), 0, mask.shape[0] - 1)
    inside = mask[y, x] > 0
    return np.where(inside)[0]


def apply_mask(img, mask):
    """Apply mask to image."""
    if mask is None or img is None:
        return img
    masked = img.copy()
    if img.ndim == 3:
        for i in range(img.shape[2]):
            masked[..., i][mask == 0] = 0
    else:
        masked[mask == 0] = 0
    return masked


def smooth_dxdy(dxdy, sigma_ratio=0.005):
    """Smooth displacement field with Gaussian filter."""
    dx, dy = dxdy[0], dxdy[1]
    sigma = max(dx.shape) * sigma_ratio
    from scipy.ndimage import gaussian_filter
    smoothed_dx = gaussian_filter(dx, sigma=sigma)
    smoothed_dy = gaussian_filter(dy, sigma=sigma)
    return [smoothed_dx, smoothed_dy]

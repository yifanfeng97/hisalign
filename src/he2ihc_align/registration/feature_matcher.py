"""Simplified feature matcher for HE-to-IHC registration.

Based on VALIS feature_matcher.py, trimmed to MVP.
Keeps only brute-force matching with RANSAC outlier rejection.
"""

from __future__ import annotations

import numpy as np
import cv2
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_kernels
from skimage import transform

from . import warp_tools
from . import feature_detectors

EPS = np.finfo(float).eps

RANSAC_NAME = "RANSAC"
USAC_MAGSAC_NAME = "USAC_MAGSAC"
RANSAC_DICT = {
    RANSAC_NAME: cv2.RANSAC,
    USAC_MAGSAC_NAME: cv2.USAC_MAGSAC,
}
DEFAULT_RANSAC = 7
DEFAULT_MATCH_FILTER = USAC_MAGSAC_NAME
DEFAULT_FD = feature_detectors.BriskFD

AMBIGUOUS_METRICS = set(metrics.pairwise._VALID_METRICS).intersection(
    metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS.keys()
)


def convert_distance_to_similarity(d, n_features=64):
    """Convert distance to similarity."""
    return np.exp(-d * (1 / n_features))


def convert_similarity_to_distance(s, n_features=64):
    """Convert similarity to distance."""
    return -np.log(s + EPS) / (1 / n_features)


def filter_matches_ransac(kp1_xy, kp2_xy, ransac_val=DEFAULT_RANSAC, method=USAC_MAGSAC_NAME):
    """Remove poor matches using RANSAC."""
    method_num = RANSAC_DICT[method]
    if kp1_xy.shape[0] >= 4:
        _, mask = cv2.findHomography(kp1_xy, kp2_xy, method_num, ransac_val)
        good_idx = np.where(mask.reshape(-1) == 1)[0]
        filtered_src = kp1_xy[good_idx, :]
        filtered_dst = kp2_xy[good_idx, :]
    else:
        filtered_src = kp1_xy.copy()
        filtered_dst = kp2_xy.copy()
        good_idx = np.arange(0, kp1_xy.shape[0])

    return filtered_src, filtered_dst, good_idx


def filter_matches_tukey(src_xy, dst_xy, tform=transform.SimilarityTransform()):
    """Detect and remove outliers using Tukey's method."""
    if len(src_xy) < 3:
        return src_xy, dst_xy, np.arange(len(src_xy))
    tform = transform.SimilarityTransform.from_estimate(src=dst_xy, dst=src_xy)
    M = tform.params
    warped_xy = warp_tools.warp_xy(src_xy, M)
    d = warp_tools.calc_d(warped_xy, dst_xy)

    q1 = np.quantile(d, 0.25)
    q3 = np.quantile(d, 0.75)
    iqr = q3 - q1
    outer_fence = 3 * iqr
    outer_fence_le = q1 - outer_fence
    outer_fence_ue = q3 + outer_fence

    inliers = [i for i, v in enumerate(d) if outer_fence_le <= v <= outer_fence_ue]
    return src_xy[inliers, :], dst_xy[inliers, :], inliers


def filter_matches(kp1_xy, kp2_xy, method=DEFAULT_MATCH_FILTER, filtering_kwargs=None):
    """Use RANSAC to remove poor matches, then Tukey."""
    if filtering_kwargs is None:
        filtering_kwargs = {}
    if method.upper() in RANSAC_DICT.keys():
        filtered_src, filtered_dst, good_idx = filter_matches_ransac(
            kp1_xy, kp2_xy, **filtering_kwargs
        )
    else:
        filtered_src, filtered_dst, good_idx = kp1_xy.copy(), kp2_xy.copy(), np.arange(len(kp1_xy))

    # Additional Tukey filtering
    filtered_src, filtered_dst, good_idx = filter_matches_tukey(filtered_src, filtered_dst)
    return filtered_src, filtered_dst, good_idx


class MatchInfo:
    """Stores match results between two images."""

    def __init__(self, matched_kp1_xy, matched_kp2_xy, distances, similarities, n_matches):
        self.matched_kp1_xy = matched_kp1_xy
        self.matched_kp2_xy = matched_kp2_xy
        self.distances = distances
        self.similarities = similarities
        self.n_matches = n_matches
        self.src_name = None
        self.dst_name = None

    def set_names(self, src_name, dst_name):
        self.src_name = src_name
        self.dst_name = dst_name


def match_descriptors(descriptors1, descriptors2, metric=None, metric_type=None,
                      p=2, max_distance=np.inf, cross_check=True, max_ratio=1.0,
                      metric_kwargs=None):
    """Brute-force matching of descriptors using sklearn."""
    if metric_kwargs is None:
        metric_kwargs = {}

    if descriptors1.shape[1] != descriptors2.shape[1]:
        raise ValueError("Descriptor length must equal.")

    if metric is None:
        if np.issubdtype(descriptors1.dtype, np.bool_):
            metric = 'hamming'
        else:
            metric = 'euclidean'

    if metric == 'minkowski':
        metric_kwargs['p'] = p

    if callable(metric) or metric in metrics.pairwise._VALID_METRICS:
        distances = metrics.pairwise_distances(descriptors1, descriptors2, metric=metric, **metric_kwargs)
        if callable(metric) and metric_type is None:
            metric_type = "distance"
        if metric_type == "similarity":
            distances = convert_similarity_to_distance(distances, n_features=descriptors1.shape[1])

    if metric in metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS:
        similarities = pairwise_kernels(descriptors1, descriptors2, metric=metric, **metric_kwargs)
        distances = convert_similarity_to_distance(similarities, n_features=descriptors1.shape[1])

    indices1 = np.arange(descriptors1.shape[0])
    indices2 = np.argmin(distances, axis=1)

    if cross_check:
        matches1 = np.argmin(distances, axis=0)
        mask = indices1 == matches1[indices2]
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_distance < np.inf:
        mask = distances[indices1, indices2] < max_distance
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_ratio < 1.0:
        best_distances = distances[indices1, indices2]
        distances[indices1, indices2] = np.inf
        second_best_indices2 = np.argmin(distances[indices1], axis=1)
        second_best_distances = distances[indices1, second_best_indices2]
        second_best_distances[second_best_distances == 0] = np.finfo(np.double).eps
        ratio = best_distances / second_best_distances
        mask = ratio < max_ratio
        indices1 = indices1[mask]
        indices2 = indices2[mask]
        return np.column_stack((indices1, indices2)), best_distances[mask], metric, metric_type

    return np.column_stack((indices1, indices2)), distances[indices1, indices2], metric, metric_type


def match_desc_and_kp(desc1, kp1_xy, desc2, kp2_xy, metric=None,
                      metric_type=None, metric_kwargs=None, max_ratio=1.0,
                      filter_method=DEFAULT_MATCH_FILTER, filtering_kwargs=None):
    """Match descriptors and keypoints, then filter outliers."""
    if filtering_kwargs is None:
        filtering_kwargs = {}

    cross_check = filter_method.upper() in RANSAC_DICT.keys()

    matches, match_distances, metric_name, metric_type = match_descriptors(
        desc1, desc2, metric=metric, metric_type=metric_type,
        metric_kwargs=metric_kwargs, max_ratio=max_ratio, cross_check=cross_check
    )

    desc1_match_idx = matches[:, 0]
    matched_kp1_xy = kp1_xy[desc1_match_idx, :]
    matched_desc1 = desc1[desc1_match_idx, :]

    desc2_match_idx = matches[:, 1]
    matched_kp2_xy = kp2_xy[desc2_match_idx, :]
    matched_desc2 = desc2[desc2_match_idx, :]

    mean_unfiltered_distance = np.mean(match_distances) if len(match_distances) > 0 else 0
    mean_unfiltered_similarity = np.mean(convert_distance_to_similarity(match_distances, n_features=desc1.shape[1])) if len(match_distances) > 0 else 0

    n_matches = len(matches)
    match_info12 = MatchInfo(
        matched_kp1_xy=matched_kp1_xy,
        matched_kp2_xy=matched_kp2_xy,
        distances=match_distances,
        similarities=convert_distance_to_similarity(match_distances, n_features=desc1.shape[1]),
        n_matches=n_matches,
    )
    match_info21 = MatchInfo(
        matched_kp1_xy=matched_kp2_xy,
        matched_kp2_xy=matched_kp1_xy,
        distances=match_distances,
        similarities=convert_distance_to_similarity(match_distances, n_features=desc1.shape[1]),
        n_matches=n_matches,
    )

    # Filter matches
    if n_matches > 0 and filter_method.upper() in RANSAC_DICT.keys():
        filtered_kp1, filtered_kp2, good_idx = filter_matches(
            matched_kp1_xy, matched_kp2_xy, method=filter_method, filtering_kwargs=filtering_kwargs
        )
        filtered_distances = match_distances[good_idx] if len(good_idx) > 0 else np.array([])
        filtered_similarities = convert_distance_to_similarity(filtered_distances, n_features=desc1.shape[1]) if len(filtered_distances) > 0 else np.array([])
    else:
        filtered_kp1 = matched_kp1_xy
        filtered_kp2 = matched_kp2_xy
        filtered_distances = match_distances
        filtered_similarities = convert_distance_to_similarity(match_distances, n_features=desc1.shape[1])

    n_filtered = len(filtered_kp1)
    filtered_match_info12 = MatchInfo(
        matched_kp1_xy=filtered_kp1,
        matched_kp2_xy=filtered_kp2,
        distances=filtered_distances,
        similarities=filtered_similarities,
        n_matches=n_filtered,
    )
    filtered_match_info21 = MatchInfo(
        matched_kp1_xy=filtered_kp2,
        matched_kp2_xy=filtered_kp1,
        distances=filtered_distances,
        similarities=filtered_similarities,
        n_matches=n_filtered,
    )

    return match_info12, filtered_match_info12, match_info21, filtered_match_info21


class Matcher:
    """Base matcher class."""

    def __init__(self, feature_detector=None, match_filter_method=DEFAULT_MATCH_FILTER):
        if feature_detector is None:
            feature_detector = DEFAULT_FD()
        self.feature_detector = feature_detector
        self.match_filter_method = match_filter_method

    def match_images(self, img1, desc1, kp1_xy, img2, desc2, kp2_xy,
                     additional_filtering_kwargs=None, sorting_images=False):
        """Match features between two images."""
        if additional_filtering_kwargs is None:
            additional_filtering_kwargs = {}

        filtering_kwargs = {"ransac_val": DEFAULT_RANSAC, "method": self.match_filter_method}
        filtering_kwargs.update(additional_filtering_kwargs)

        return match_desc_and_kp(
            desc1, kp1_xy, desc2, kp2_xy,
            filter_method=self.match_filter_method,
            filtering_kwargs=filtering_kwargs,
        )

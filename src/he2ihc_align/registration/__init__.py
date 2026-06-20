"""Registration package for HE-to-IHC whole-slide image alignment."""

from he2ihc_align.registration.feature_detectors import (
    AkazeFD,
    BriskFD,
    FeatureDD,
    KazeFD,
    OrbFD,
    VggFD,
)
from he2ihc_align.registration.feature_matcher import (
    Matcher,
    MatchInfo,
    filter_matches,
    match_descriptors,
)
from he2ihc_align.registration.non_rigid import (
    NonRigidRegistrar,
    NonRigidRegistrarBase,
    NonRigidRegistrarXY,
    OpticalFlowWarper,
    SimpleElastixWarper,
)
from he2ihc_align.registration.registrar import HEIHCRegistrar
from he2ihc_align.registration.warp_tools import (
    calc_d,
    get_shape,
    rescale_img,
    resize_img,
    warp_img,
    warp_xy,
)

__all__ = [
    "HEIHCRegistrar",
    "NonRigidRegistrar",
    "NonRigidRegistrarBase",
    "NonRigidRegistrarXY",
    "OpticalFlowWarper",
    "SimpleElastixWarper",
    "MatchInfo",
    "Matcher",
    "filter_matches",
    "match_descriptors",
    "FeatureDD",
    "BriskFD",
    "KazeFD",
    "AkazeFD",
    "OrbFD",
    "VggFD",
    "warp_xy",
    "warp_img",
    "get_shape",
    "rescale_img",
    "resize_img",
    "calc_d",
]

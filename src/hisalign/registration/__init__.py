"""Registration package for HE-to-IHC whole-slide image alignment."""

from hisalign.registration.feature_detectors import (
    AkazeFD,
    BriskFD,
    FeatureDD,
    KazeFD,
    OrbFD,
    VggFD,
)
from hisalign.registration.feature_matcher import (
    Matcher,
    MatchInfo,
    filter_matches,
    match_descriptors,
)
from hisalign.registration.non_rigid import (
    NonRigidRegistrar,
    NonRigidRegistrarBase,
    NonRigidRegistrarXY,
    OpticalFlowWarper,
    SimpleElastixWarper,
)
from hisalign.registration.registrar import HEIHCRegistrar
from hisalign.registration.warp_tools import (
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

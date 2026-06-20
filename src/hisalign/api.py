"""Public API for hisalign.

Provides the high-level `HisAlign` class and the serializable `HisAlignModel`
for offline coordinate mapping between H&E and IHC whole-slide images.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hisalign.registration import feature_detectors, feature_matcher, warp_tools
from hisalign.registration.registrar import HEIHCRegistrar
from hisalign.slide_io.factory import get_slide_mpp, open_slide

HISALIGN_VERSION = "0.2.0"


@dataclass
class HisAlignModel:
    """Serializable alignment model for offline coordinate mapping.

    All coordinate mapping uses level-0 pixel coordinates. The stored scale
    factors convert between level-0 pixels and the processed registration-image
    pixels on which the transforms were estimated.

    Attributes
    ----------
    version : str
        Package version used to create the model.
    he_path : str
        Absolute path to the HE reference slide.
    ihc_paths : dict[str, str]
        Mapping of marker names to absolute IHC slide file paths.
    config : dict
        Normalized configuration used during registration.
    he_scale_to_level0 : float
        Scale factor converting one processed HE registration pixel to a
        level-0 HE pixel.
    ihc_scale_to_level0 : dict[str, float]
        Per-marker scale factors from processed IHC registration pixels to
        level-0 IHC pixels.
    he_padding_matrix : np.ndarray
        3x3 affine matrix that pads the processed HE image to the common canvas.
    ihc_padding_matrix : dict[str, np.ndarray]
        Per-marker 3x3 affine padding matrices.
    rigid_matrix : dict[str, np.ndarray]
        Per-marker 3x3 rigid transformation matrices (moving → reference).
    bk_dxdy : dict[str, tuple[np.ndarray, np.ndarray] | None]
        Per-marker backward displacement fields, or ``None`` if not available.
    fwd_dxdy : dict[str, tuple[np.ndarray, np.ndarray] | None]
        Per-marker forward displacement fields, or ``None`` if not available.
    mpp : float | None
        Microns per pixel at level 0 of the HE slide.
    reg_shape_rc : dict[str, tuple[int, int]]
        Common registration canvas shape for each marker.
    """

    version: str = HISALIGN_VERSION
    he_path: str = ""
    ihc_paths: dict[str, str] = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    he_scale_to_level0: float = 1.0
    ihc_scale_to_level0: dict[str, float] = field(default_factory=dict)
    he_padding_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    ihc_padding_matrix: dict[str, np.ndarray] = field(default_factory=dict)
    rigid_matrix: dict[str, np.ndarray] = field(default_factory=dict)
    bk_dxdy: dict[str, tuple[np.ndarray, np.ndarray] | None] = field(
        default_factory=dict
    )
    fwd_dxdy: dict[str, tuple[np.ndarray, np.ndarray] | None] = field(
        default_factory=dict
    )
    mpp: float | None = None
    reg_shape_rc: dict[str, tuple[int, int]] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Serialize the model to a ``.pkl`` file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "HisAlignModel":
        """Deserialize a model from a ``.pkl`` file."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj

    def warp_xy(
        self,
        coords: np.ndarray,
        marker: str,
        direction: str = "he_to_ihc",
    ) -> np.ndarray:
        """Map level-0 coordinates between HE and IHC spaces offline.

        Parameters
        ----------
        coords : np.ndarray
            (N, 2) array of level-0 pixel coordinates in ``(x, y)`` order.
        marker : str
            Marker name.
        direction : str
            Either ``"he_to_ihc"`` or ``"ihc_to_he"``.

        Returns
        -------
        np.ndarray
            (N, 2) array of mapped level-0 pixel coordinates.
        """
        if marker not in self.rigid_matrix:
            raise KeyError(
                f"Marker '{marker}' not found. Available: {list(self.rigid_matrix.keys())}"
            )

        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        he_scale = self.he_scale_to_level0
        ihc_scale = self.ihc_scale_to_level0[marker]
        ihc_padding_inv = np.linalg.inv(self.ihc_padding_matrix[marker])
        he_padding_inv = np.linalg.inv(self.he_padding_matrix)
        inv_rigid_matrix = np.linalg.inv(self.rigid_matrix[marker])

        if direction == "he_to_ihc":
            xy_reg = coords / he_scale
            xy_padded = warp_tools.warp_xy(xy_reg, M=self.he_padding_matrix)
            fwd = self.fwd_dxdy.get(marker)
            if fwd is not None:
                xy_padded = warp_tools.warp_xy(xy_padded, fwd_dxdy=fwd)
            ihc_padded = warp_tools.warp_xy(xy_padded, M=inv_rigid_matrix)
            ihc_reg = warp_tools.warp_xy(ihc_padded, M=ihc_padding_inv)
            return ihc_reg * ihc_scale

        if direction == "ihc_to_he":
            xy_reg = coords / ihc_scale
            xy_padded = warp_tools.warp_xy(xy_reg, M=self.ihc_padding_matrix[marker])
            bk = self.bk_dxdy.get(marker)
            if bk is not None:
                xy_padded = warp_tools.warp_xy(xy_padded, bk_dxdy=bk)
            he_padded = warp_tools.warp_xy(xy_padded, M=self.rigid_matrix[marker])
            he_reg = warp_tools.warp_xy(he_padded, M=he_padding_inv)
            return he_reg * he_scale

        raise ValueError(
            f"direction must be 'he_to_ihc' or 'ihc_to_he', got {direction!r}"
        )


def warp_xy(
    coords: np.ndarray,
    marker: str,
    direction: str,
    model: HisAlignModel,
) -> np.ndarray:
    """Standalone coordinate mapping using a loaded ``HisAlignModel``."""
    return model.warp_xy(coords, marker, direction)


class HisAlign:
    """High-level API for registering IHC slides to an H&E reference slide.

    Parameters
    ----------
    he_path : str | Path
        Path to the HE reference slide.
    ihc_paths : dict[str, str | Path] | list[str | Path]
        Either a dict mapping marker names to file paths, or a list of paths
        from which marker names are derived from the filename stem.
    registration_level : int, default 3
        Pyramid level used for registration.
    max_image_dim_px : int, default 1024
        Maximum dimension of the processed registration image.
    preprocessing : str, default "od"
        Preprocessing method: ``"od"`` (optical density) or ``"gray"``.
    feature_detector : str, default "kaze"
        Feature detector name, e.g. ``"kaze"``, ``"akaze"``, ``"sift"``,
        ``"orb"``, ``"brisk"``.
    feature_n_levels : int, default 3
        Number of feature pyramid levels.
    match_max_ratio : float, default 1.0
        Lowe's ratio-test threshold; ``1.0`` disables the ratio test.
    mpp : float | None, default None
        Optional explicit microns-per-pixel value. If ``None``, the value is
        read from the HE slide metadata when available.
    """

    def __init__(
        self,
        he_path: str | Path,
        ihc_paths: dict[str, str | Path] | list[str | Path],
        registration_level: int = 3,
        max_image_dim_px: int = 1024,
        preprocessing: str = "od",
        feature_detector: str = "kaze",
        feature_n_levels: int = 3,
        match_max_ratio: float = 1.0,
        mpp: float | None = None,
    ):
        self.he_path = Path(he_path).resolve()
        self.ihc_paths = self._normalize_ihc_paths(ihc_paths)
        self.registration_level = registration_level
        self.max_image_dim_px = max_image_dim_px
        self.preprocessing = preprocessing
        self.feature_detector_name = feature_detector
        self.feature_n_levels = feature_n_levels
        self.match_max_ratio = match_max_ratio
        self._mpp = mpp
        self._registrar: HEIHCRegistrar | None = None

    @staticmethod
    def _normalize_ihc_paths(
        ihc_paths: dict[str, str | Path] | list[str | Path],
    ) -> dict[str, Path]:
        """Normalize IHC input to a ``{marker: path}`` mapping."""
        if isinstance(ihc_paths, dict):
            return {k: Path(v).resolve() for k, v in ihc_paths.items()}

        result: dict[str, Path] = {}
        for p in ihc_paths:
            path = Path(p).resolve()
            parts = path.stem.split()
            marker = parts[-1] if parts else path.stem
            if marker in result:
                raise ValueError(f"Duplicate marker name '{marker}' from {path}")
            result[marker] = path
        return result

    def fit(self) -> HisAlignModel:
        """Open slides, run registration, and return a serializable model.

        Returns
        -------
        HisAlignModel
            Model containing all transforms for offline coordinate mapping.
        """
        he_slide = open_slide(self.he_path)
        ihc_slides = {
            marker: open_slide(path) for marker, path in self.ihc_paths.items()
        }

        try:
            mpp = self._mpp
            if mpp is None:
                mpp = get_slide_mpp(he_slide)

            feature_detector = feature_detectors.create_feature_detector(
                self.feature_detector_name,
                n_levels=self.feature_n_levels,
            )
            matcher = feature_matcher.Matcher(
                feature_detector=feature_detector,
                max_ratio=self.match_max_ratio,
            )

            registrar = HEIHCRegistrar(
                he_slide=he_slide,
                ihc_slides=ihc_slides,
                registration_level=self.registration_level,
                max_image_dim_px=self.max_image_dim_px,
                preprocessing_method=self.preprocessing,
                feature_detector=feature_detector,
                matcher=matcher,
            )
            registrar.fit()
            self._registrar = registrar

            model = HisAlignModel(
                version=HISALIGN_VERSION,
                he_path=str(self.he_path),
                ihc_paths={k: str(v) for k, v in self.ihc_paths.items()},
                config={
                    "registration_level": self.registration_level,
                    "max_image_dim_px": self.max_image_dim_px,
                    "preprocessing": self.preprocessing,
                    "feature_detector": self.feature_detector_name,
                    "feature_n_levels": self.feature_n_levels,
                    "match_max_ratio": self.match_max_ratio,
                    "mpp": mpp,
                },
                he_scale_to_level0=registrar.he_scale_to_level0,
                ihc_scale_to_level0=registrar.ihc_scale_to_level0,
                he_padding_matrix=registrar.he_padding_matrix,
                ihc_padding_matrix=registrar.ihc_padding_matrix,
                rigid_matrix={
                    marker: registrar.rigid_registrars[marker].M
                    for marker in registrar.rigid_registrars
                },
                bk_dxdy={
                    marker: (
                        (nr.bk_dxdy[0], nr.bk_dxdy[1])
                        if nr.bk_dxdy is not None
                        else None
                    )
                    for marker, nr in registrar.non_rigid_registrars.items()
                },
                fwd_dxdy={
                    marker: (
                        (nr.fwd_dxdy[0], nr.fwd_dxdy[1])
                        if nr.fwd_dxdy is not None
                        else None
                    )
                    for marker, nr in registrar.non_rigid_registrars.items()
                },
                mpp=mpp,
                reg_shape_rc=registrar.reg_shape_rc,
            )
            return model
        finally:
            he_slide.close()
            for slide in ihc_slides.values():
                slide.close()

    def warp_xy_from_he_to_ihc(
        self,
        xy: np.ndarray,
        marker: str,
    ) -> np.ndarray:
        """Map level-0 HE coordinates to level-0 IHC coordinates."""
        if self._registrar is None:
            raise RuntimeError("fit() must be called before coordinate mapping")
        return self._registrar.warp_xy_from_he_to_ihc(xy, marker)

    def warp_xy_from_ihc_to_he(
        self,
        xy: np.ndarray,
        marker: str,
    ) -> np.ndarray:
        """Map level-0 IHC coordinates to level-0 HE coordinates."""
        if self._registrar is None:
            raise RuntimeError("fit() must be called before coordinate mapping")
        return self._registrar.warp_xy_from_ihc_to_he(xy, marker)

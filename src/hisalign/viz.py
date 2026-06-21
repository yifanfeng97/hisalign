"""Visualization utilities for HE-to-IHC alignment gallery and report."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from skimage import color as skcolor

from hisalign.registration import warp_tools
from hisalign.slide_io.base import Slide

if TYPE_CHECKING:
    from hisalign.registration import non_rigid, rigid


def read_patch_rgb(
    slide: Slide,
    x: int,
    y: int,
    w: int,
    h: int,
    level: int = 0,
) -> np.ndarray:
    """Read a region from a slide, clamp to bounds, and resize to requested size.

    Parameters
    ----------
    slide : Slide
        Slide to read from.
    x, y : int
        Top-left corner in level-0 coordinates.
    w, h : int
        Requested width and height in level-0 coordinates.
    level : int, default 0
        Pyramid level to read from.

    Returns
    -------
    np.ndarray
        HWC uint8 RGB array of shape (h, w, 3).
    """
    # Clamp to slide bounds
    level0_w, level0_h = slide.level_dimensions[0]
    x = max(0, min(x, level0_w))
    y = max(0, min(y, level0_h))
    w = min(w, level0_w - x)
    h = min(h, level0_h - y)

    if w <= 0 or h <= 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    downsample = slide.level_downsamples[level]
    # Read at the requested level with size scaled accordingly
    level_w = max(1, int(w / downsample))
    level_h = max(1, int(h / downsample))

    patch = slide.read_region((x, y), level, (level_w, level_h))

    # Resize to the exact requested size if needed
    if patch.shape[0] != h or patch.shape[1] != w:
        from PIL import Image

        pil_img = Image.fromarray(patch)
        pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)
        patch = np.array(pil_img)

    return patch


def sample_patch_indices(
    n_patches: int,
    viz_sample_n: int,
    random_seed: int | None,
    clipped_flags: list[bool] | None = None,
    include_clipped: bool = True,
) -> list[int]:
    """Randomly sample patch indices for visualization.

    Parameters
    ----------
    n_patches : int
        Total number of patches available.
    viz_sample_n : int
        Number of patches to sample.
    random_seed : int | None
        Seed for reproducibility. If None, sampling is non-deterministic.
    clipped_flags : list[bool] | None
        Optional per-patch clipped flags. If provided and ``include_clipped``
        is False, clipped patches are excluded from sampling.
    include_clipped : bool
        Whether to allow sampling of clipped patches.

    Returns
    -------
    list[int]
        List of sampled patch indices.
    """
    rng = np.random.default_rng(random_seed)
    candidates = list(range(n_patches))

    if clipped_flags is not None and not include_clipped:
        candidates = [i for i, clipped in enumerate(clipped_flags) if not clipped]

    if len(candidates) == 0:
        return []

    n = min(viz_sample_n, len(candidates))
    sampled = rng.choice(candidates, size=n, replace=False)
    return [int(i) for i in sampled]


def make_patch_figure(
    he_patch: np.ndarray,
    ihc_patches: dict[str, np.ndarray],
    title: str,
    clipped_flags: dict[str, bool] | None = None,
) -> plt.Figure:
    """Create a matplotlib figure with HE + all IHC marker patches.

    Parameters
    ----------
    he_patch : np.ndarray
        HE patch image.
    ihc_patches : dict[str, np.ndarray]
        Dictionary mapping marker name to IHC patch image.
    title : str
        Figure title.
    clipped_flags : dict[str, bool] | None
        Optional dict indicating whether each marker's IHC patch is clipped.
        Clipped patches get a red badge in the top-left corner.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    n_markers = len(ihc_patches)
    n_cols = n_markers + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(he_patch)
    axes[0].set_title("HE")
    axes[0].axis("off")

    clipped_flags = clipped_flags or {}
    for idx, (marker, patch) in enumerate(ihc_patches.items(), start=1):
        if patch.size == 0:
            patch = np.zeros_like(he_patch)
        axes[idx].imshow(patch)
        axes[idx].set_title(marker)
        axes[idx].axis("off")

        if clipped_flags.get(marker, False):
            axes[idx].text(
                0.02,
                0.98,
                "CLIPPED",
                transform=axes[idx].transAxes,
                fontsize=8,
                color="red",
                weight="bold",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    return fig


def fig_to_data_uri(
    fig: plt.Figure,
    fmt: str = "png",
    dpi: int = 100,
    quality: int | None = None,
) -> str:
    """Convert a matplotlib figure to a base64-encoded image data URI.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure.
    fmt : str, default "png"
        Output image format (e.g. "png" or "jpeg").
    dpi : int, default 100
        Resolution passed to ``savefig``.
    quality : int | None, default None
        JPEG/WebP quality (0-100). Ignored for PNG.

    Returns
    -------
    str
        Base64 data URI string.
    """
    buf = io.BytesIO()
    kwargs: dict[str, Any] = {"format": fmt, "bbox_inches": "tight", "dpi": dpi}
    if fmt in ("jpeg", "webp") and quality is not None:
        kwargs["pil_kwargs"] = {"quality": quality}
    fig.savefig(buf, **kwargs)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    mime = "image/jpeg" if fmt == "jpeg" else ("image/webp" if fmt == "webp" else "image/png")
    return f"data:{mime};base64,{data}"


def create_html_gallery(
    output_path: Path,
    slide_id: str,
    entries: list[dict],
) -> Path:
    """Write a minimal HTML page with embedded base64 PNGs.

    Parameters
    ----------
    output_path : Path
        Path to write the HTML file.
    slide_id : str
        Slide identifier for the page title.
    entries : list[dict]
        List of dicts with keys ``title`` and ``data_uri``.

    Returns
    -------
    Path
        Path to the written HTML file.
    """
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '  <meta charset="UTF-8">',
        f"  <title>Gallery - {html.escape(slide_id)}</title>",
        "  <style>",
        "    body { font-family: sans-serif; margin: 20px; }",
        "    .entry { margin-bottom: 40px; }",
        "    .entry img { max-width: 100%; border: 1px solid #ccc; }",
        "    .entry h3 { margin-bottom: 5px; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>Gallery - {html.escape(slide_id)}</h1>",
    ]

    for entry in entries:
        title = html.escape(entry["title"])
        html_parts.append('  <div class="entry">')
        html_parts.append(f"    <h3>{title}</h3>")
        html_parts.append(f'    <img src="{entry["data_uri"]}" alt="{title}">')
        html_parts.append("  </div>")

    html_parts.extend(["</body>", "</html>"])

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    return output_path


def read_slide_thumbnail(
    slide: Slide,
    level: int,
    max_dim_px: int | None = None,
) -> np.ndarray:
    """Read a whole slide at a pyramid level, optionally resizing.

    Parameters
    ----------
    slide : Slide
        Slide to read.
    level : int
        Pyramid level.
    max_dim_px : int | None
        If set, resize so the longest side is at most this many pixels.

    Returns
    -------
    np.ndarray
        HWC uint8 RGB thumbnail.
    """
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h))

    if max_dim_px is not None and max(w, h) > max_dim_px:
        scale = max_dim_px / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = warp_tools.resize_img(img, (new_h, new_w))

    return img


def read_global_thumbnail(
    slide: Slide,
    max_dim_px: int = 2048,
) -> tuple[np.ndarray, float]:
    """Read a whole-slide HE thumbnail and return its level-0 downsample factor.

    Parameters
    ----------
    slide : Slide
        HE reference slide.
    max_dim_px : int, default 2048
        Longest side of the returned thumbnail.

    Returns
    -------
    tuple[np.ndarray, float]
        HWC uint8 RGB thumbnail and the downsample factor relative to level 0.
    """
    level0_w, level0_h = slide.level_dimensions[0]
    target_downsample = max(level0_w, level0_h) / max_dim_px
    level = slide.get_best_level_for_downsample(target_downsample)
    thumb = read_slide_thumbnail(slide, level=level, max_dim_px=max_dim_px)
    thumb_h, thumb_w = thumb.shape[0:2]
    actual_downsample = max(level0_w, level0_h) / max(thumb_w, thumb_h)
    return thumb, actual_downsample


def make_patch_with_context_figure(
    he_global_thumb: np.ndarray,
    he_downsample: float,
    he_bbox_xywh: tuple[int, int, int, int],
    ihc_global_thumbs: dict[str, np.ndarray],
    ihc_downsamples: dict[str, float],
    ihc_bboxes_xywh: dict[str, tuple[int, int, int, int]],
    he_patch: np.ndarray,
    ihc_patches: dict[str, np.ndarray],
    title: str,
    clipped_flags: dict[str, bool] | None = None,
    col_inches: float = 4.0,
) -> plt.Figure:
    """Create a combined figure: per-marker global context + local patches.

    Layout:
      - Row 0: HE global thumbnail with patch box, then each marker's global
        IHC thumbnail with the corresponding registered patch box.
      - Row 1: Local HE patch, then each marker's local registered IHC patch.

    Parameters
    ----------
    he_global_thumb : np.ndarray
        HWC uint8 RGB whole-slide HE thumbnail.
    he_downsample : float
        HE thumbnail downsample factor relative to level 0.
    he_bbox_xywh : tuple[int, int, int, int]
        HE patch bounding box in level-0 pixels (x, y, w, h).
    ihc_global_thumbs : dict[str, np.ndarray]
        Per-marker HWC uint8 RGB whole-slide IHC thumbnails.
    ihc_downsamples : dict[str, float]
        Per-marker IHC thumbnail downsample factors.
    ihc_bboxes_xywh : dict[str, tuple[int, int, int, int]]
        Per-marker registered IHC patch bounding boxes in level-0 pixels.
    he_patch : np.ndarray
        Local HE patch.
    ihc_patches : dict[str, np.ndarray]
        Per-marker registered IHC patches.
    title : str
        Figure title.
    clipped_flags : dict[str, bool] | None
        Optional per-marker clipped badges for local IHC patches.
    col_inches : float, default 4.0
        Width in inches allocated to each marker column; total figure width is
        ``col_inches * n_cols``.

    Returns
    -------
    plt.Figure
        Matplotlib figure with global context on top and local patches below.
    """
    n_markers = len(ihc_patches)
    n_cols = n_markers + 1

    fig = plt.figure(figsize=(col_inches * n_cols, 8))
    gs = fig.add_gridspec(2, n_cols, height_ratios=[1, 1])

    def _add_thumb_with_box(ax, thumb, downsample, bbox_xywh, label):
        ax.imshow(thumb)
        x, y, w, h = bbox_xywh
        rect = patches.Rectangle(
            (x / downsample, y / downsample),
            w / downsample,
            h / downsample,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.set_title(label)
        ax.axis("off")

    # Row 0: global thumbnails with boxes
    ax_he_global = fig.add_subplot(gs[0, 0])
    _add_thumb_with_box(
        ax_he_global, he_global_thumb, he_downsample, he_bbox_xywh, "HE global"
    )

    for col_idx, marker in enumerate(ihc_patches.keys(), start=1):
        ax = fig.add_subplot(gs[0, col_idx])
        _add_thumb_with_box(
            ax,
            ihc_global_thumbs[marker],
            ihc_downsamples[marker],
            ihc_bboxes_xywh[marker],
            f"{marker} global",
        )

    # Row 1: local patches
    ax_he = fig.add_subplot(gs[1, 0])
    ax_he.imshow(he_patch)
    ax_he.set_title("HE")
    ax_he.axis("off")

    clipped_flags = clipped_flags or {}
    for col_idx, (marker, patch) in enumerate(ihc_patches.items(), start=1):
        if patch.size == 0:
            patch = np.zeros_like(he_patch)
        ax = fig.add_subplot(gs[1, col_idx])
        ax.imshow(patch)
        ax.set_title(marker)
        ax.axis("off")

        if clipped_flags.get(marker, False):
            ax.text(
                0.02,
                0.98,
                "CLIPPED",
                transform=ax.transAxes,
                fontsize=8,
                color="red",
                weight="bold",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    fig.tight_layout()
    return fig


def _resize_to_match(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a HWC image to match (height, width) of target_shape."""
    if img.shape[:2] == target_shape[:2]:
        return img
    return warp_tools.resize_img(img, target_shape[:2])


def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    """Convert an RGB image to a uint8 grayscale image."""
    if img.ndim == 3 and img.shape[2] == 3:
        gray = skcolor.rgb2gray(img)
    else:
        gray = img.astype(np.float64)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8) * 255
    return gray.astype(np.uint8)


def _make_overlay_array(he_img: np.ndarray, ihc_img: np.ndarray) -> np.ndarray:
    """Create a green/magenta overlay array (HE=green, IHC=magenta)."""
    ihc_img = _resize_to_match(ihc_img, he_img.shape)

    he_gray = _to_gray_uint8(he_img)
    ihc_gray = _to_gray_uint8(ihc_img)

    overlay = np.zeros((he_gray.shape[0], he_gray.shape[1], 3), dtype=np.uint8)
    overlay[..., 1] = he_gray  # green channel for HE
    overlay[..., 0] = ihc_gray  # red channel for IHC
    overlay[..., 2] = ihc_gray  # blue channel for IHC
    return overlay


def make_overlay_figure(
    he_img: np.ndarray,
    ihc_img: np.ndarray,
    title: str,
) -> plt.Figure:
    """Create a green/magenta overlay figure before registration."""
    overlay = _make_overlay_array(he_img, ihc_img)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def make_rigid_overlay_figure(
    he_img: np.ndarray,
    ihc_img: np.ndarray,
    m: np.ndarray,
    title: str,
) -> plt.Figure:
    """Create a green/magenta overlay figure after rigid registration."""
    he_gray = _to_gray_uint8(he_img)
    ihc_gray = _to_gray_uint8(ihc_img)

    # Warp IHC to HE space using rigid transform
    from hisalign.registration.rigid import RigidRegistrar

    rigid_reg = RigidRegistrar(
        ref_img=he_gray,
        moving_img=ihc_gray,
    )
    rigid_reg.M = m
    warped_ihc = rigid_reg.warp_image(ihc_gray, out_shape_rc=he_gray.shape)

    overlay = np.zeros((he_gray.shape[0], he_gray.shape[1], 3), dtype=np.uint8)
    overlay[..., 1] = he_gray
    overlay[..., 0] = warped_ihc
    overlay[..., 2] = warped_ihc

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def make_non_rigid_overlay_figure(
    he_img: np.ndarray,
    ihc_img: np.ndarray,
    non_rigid_registrar: non_rigid.NonRigidRegistrar,
    title: str,
) -> plt.Figure:
    """Create a green/magenta overlay figure after non-rigid registration."""
    he_gray = _to_gray_uint8(he_img)
    ihc_gray = _to_gray_uint8(ihc_img)

    warped_ihc = non_rigid_registrar.warp_image(ihc_gray, out_shape_rc=he_gray.shape)

    overlay = np.zeros((he_gray.shape[0], he_gray.shape[1], 3), dtype=np.uint8)
    overlay[..., 1] = he_gray
    overlay[..., 0] = warped_ihc
    overlay[..., 2] = warped_ihc

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def make_deformation_field_figure(
    bk_dxdy: list[np.ndarray],
    title: str = "Deformation field",
    n_arrows: int = 32,
) -> plt.Figure:
    """Visualize a backward displacement field as a quiver plot."""
    dx, dy = bk_dxdy
    h, w = dx.shape

    step_r = max(1, h // n_arrows)
    step_c = max(1, w // n_arrows)
    y, x = np.mgrid[0:h:step_r, 0:w:step_c]
    u = dx[y, x]
    v = dy[y, x]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(np.sqrt(dx**2 + dy**2), cmap="viridis", aspect="auto")
    ax.quiver(x, y, u, v, color="white", alpha=0.7, scale=max(w, h) * 2)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def make_marker_thumbnail_figure(
    he_img: np.ndarray,
    ihc_img: np.ndarray,
    non_rigid_registrar: non_rigid.NonRigidRegistrar,
    title: str,
) -> plt.Figure:
    """Create a side-by-side colored thumbnail: HE and registered IHC."""
    he_img = _resize_to_match(he_img, he_img.shape)
    ihc_img = _resize_to_match(ihc_img, he_img.shape)
    warped_ihc = non_rigid_registrar.warp_image(ihc_img, out_shape_rc=he_img.shape)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(he_img)
    axes[0].set_title("HE")
    axes[0].axis("off")

    axes[1].imshow(warped_ihc)
    axes[1].set_title("Registered IHC")
    axes[1].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def make_single_thumbnail_figure(
    img: np.ndarray,
    title: str,
) -> plt.Figure:
    """Create a single-image colored thumbnail figure."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def compute_marker_metrics(
    rigid_registrar: rigid.RigidRegistrar,
    non_rigid_registrar: non_rigid.NonRigidRegistrar,
    mpp: float,
) -> dict:
    """Compute per-marker registration metrics in the common registration canvas.

    Matched keypoints are already expressed in the padded common canvas because
    both rigid and non-rigid registrars were fit on padded images.  Distances
    are therefore computed directly in that canvas and then converted to
    micrometers using ``mpp`` (the physical size of one common-canvas pixel).

    Parameters
    ----------
    rigid_registrar
        Fitted rigid registrar containing matched keypoints.
    non_rigid_registrar
        Fitted non-rigid registrar operating on the same common canvas.
    mpp
        Microns per pixel of the common registration canvas.

    Returns dict with keys:
    - original_displacement_um
    - rigid_displacement_um
    - non_rigid_displacement_um
    - rtre
    - n_matches
    """
    ref_kp = rigid_registrar.matched_kp_ref
    moving_kp = rigid_registrar.matched_kp_moving

    if ref_kp is None or moving_kp is None or len(ref_kp) == 0:
        return {
            "original_displacement_um": 0.0,
            "rigid_displacement_um": 0.0,
            "non_rigid_displacement_um": 0.0,
            "rtre": 0.0,
            "n_matches": 0,
        }

    ref_kp = np.asarray(ref_kp, dtype=np.float64)
    moving_kp = np.asarray(moving_kp, dtype=np.float64)

    reg_shape_rc = rigid_registrar.ref_img.shape[0:2]
    diagonal_px = np.sqrt(reg_shape_rc[0] ** 2 + reg_shape_rc[1] ** 2)

    original_d_px = np.median(warp_tools.calc_d(ref_kp, moving_kp))
    moving_kp_rigid = rigid_registrar.warp_xy(moving_kp)
    rigid_d_px = np.median(warp_tools.calc_d(ref_kp, moving_kp_rigid))
    moving_kp_nr = non_rigid_registrar.warp_xy(moving_kp)
    nr_d_px = np.median(warp_tools.calc_d(ref_kp, moving_kp_nr))

    original_d_um = float(original_d_px * mpp)
    rigid_d_um = float(rigid_d_px * mpp)
    nr_d_um = float(nr_d_px * mpp)

    rtre = float(
        np.median(warp_tools.calc_d(ref_kp, moving_kp_nr)) / diagonal_px * 100.0
    )

    return {
        "original_displacement_um": original_d_um,
        "rigid_displacement_um": rigid_d_um,
        "non_rigid_displacement_um": nr_d_um,
        "rtre": rtre,
        "n_matches": int(rigid_registrar.n_matches),
    }


def compute_overall_metrics(marker_metrics: dict[str, dict]) -> dict:
    """Aggregate per-marker metrics into overall statistics."""
    if not marker_metrics:
        return {
            "original_displacement_um": 0.0,
            "rigid_displacement_um": 0.0,
            "non_rigid_displacement_um": 0.0,
            "rtre": 0.0,
            "n_matches": 0,
        }

    keys = [
        "original_displacement_um",
        "rigid_displacement_um",
        "non_rigid_displacement_um",
        "rtre",
        "n_matches",
    ]
    return {k: float(np.mean([m[k] for m in marker_metrics.values()])) for k in keys}


def create_html_report(
    output_path: Path,
    slide_id: str,
    overall_metrics: dict,
    overlay_entries: list[dict],
    marker_rows: list[dict],
    report_path: Path | None = None,
    rtre_threshold: float = 5.0,
    he_ref_thumb_uri: str = "",
    gallery_entries: list[dict] | None = None,
) -> Path:
    """Write a self-contained registration quality report as HTML.

    When ``gallery_entries`` is provided, the report is rendered as a single
    unified page with tabs for the slide summary and the patch gallery.  When
    ``gallery_entries`` is ``None`` the legacy slide-only report is produced.

    Parameters
    ----------
    output_path : Path
        Path to write the HTML file.
    slide_id : str
        Slide/case identifier.
    overall_metrics : dict
        Dict with keys original_displacement_um, rigid_displacement_um,
        non_rigid_displacement_um, rtre, n_matches.
    overlay_entries : list[dict]
        List of dicts with keys ``title`` and ``data_uri``.
    marker_rows : list[dict]
        List of dicts with keys marker, original_displacement_um,
        rigid_displacement_um, non_rigid_displacement_um, rtre, n_matches,
        he_thumb_uri, ihc_thumb_uri, def_uri.
    report_path : Path, optional
        Output path shown in the report footer. Defaults to ``output_path``.
    rtre_threshold : float
        rTRE value below which is highlighted as good.
    he_ref_thumb_uri : str
        Data URI of the HE reference thumbnail shown in the table header row.
    gallery_entries : list[dict] | None
        Optional patch gallery entries; each dict has keys ``title`` and
        ``data_uri``.  When provided, a "Patch Gallery" tab is added.

    Returns
    -------
    Path
        Path to the written HTML file.
    """
    report_path = report_path or output_path
    from datetime import datetime

    unified = gallery_entries is not None

    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '  <meta charset="UTF-8">',
        f"  <title>配准报告 - {html.escape(slide_id)}</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans SC', sans-serif; margin: 2rem; background: #f7f8fa; color: #222; }",
        "    h1, h2 { color: #1a1a2e; }",
        "    .card { background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); padding: 1.5rem; margin-bottom: 2rem; }",
        "    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }",
        "    .metric { background: #f0f4ff; border-radius: 8px; padding: 1rem; text-align: center; }",
        "    .metric .value { font-size: 1.6rem; font-weight: 700; color: #2d4fda; }",
        "    .metric .label { font-size: 0.9rem; color: #555; margin-top: 0.3rem; }",
        "    .overlap-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }",
        "    .overlap-item img { width: 100%; border-radius: 8px; border: 1px solid #e1e4e8; }",
        "    .overlap-item .caption { text-align: center; margin-top: 0.5rem; font-weight: 600; color: #444; }",
        "    table { width: 100%; border-collapse: collapse; margin-top: 1rem; }",
        "    th, td { padding: 0.7rem; text-align: left; border-bottom: 1px solid #e1e4e8; }",
        "    th { background: #f0f4ff; color: #1a1a2e; }",
        "    tr:hover { background: #fafbff; }",
        "    .slide-thumb { max-width: 220px; border-radius: 6px; border: 1px solid #e1e4e8; }",
        "    .small { font-size: 0.85rem; color: #666; }",
        "    .good { color: #2a9d3a; font-weight: 600; }",
        "    .he-row { background: #f6fff6; }",
    ]

    if unified:
        html_parts.extend(
            [
                "    .tab-nav { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; border-bottom: 2px solid #e1e4e8; }",
                "    .tab-nav button { background: none; border: none; padding: 0.8rem 1.5rem; font-size: 1rem; cursor: pointer; color: #555; border-bottom: 3px solid transparent; margin-bottom: -2px; }",
                "    .tab-nav button.active { color: #2d4fda; border-bottom-color: #2d4fda; font-weight: 600; }",
                "    .tab-panel { display: none; }",
                "    .tab-panel.active { display: block; }",
                "    .gallery-card { padding: 0; background: transparent; box-shadow: none; border-radius: 0; }",
                "    .gallery-grid { display: flex; flex-direction: column; gap: 1rem; align-items: stretch; }",
                "    .gallery-item { width: 100%; max-width: none; padding: 0; background: transparent; box-shadow: none; border-radius: 0; }",
                "    .gallery-item img { width: 100%; display: block; border: none; border-radius: 0; }",
                "    .gallery-item h4 { margin: 0.5rem 0 0.2rem; font-size: 0.95rem; color: #333; text-align: center; }",
                "    .gallery-item p { margin: 0; font-size: 0.8rem; color: #666; text-align: center; }",
            ]
        )

    html_parts.extend(
        [
            "  </style>",
            "</head>",
            "<body>",
        ]
    )

    slide_parts = [
        f"  <h1>🔬 配准报告 – {html.escape(slide_id)}</h1>",
        "  <div class='card'>",
        "    <h2>整体误差统计</h2>",
        "    <div class='metrics'>",
        f"      <div class='metric'><div class='value'>{overall_metrics['original_displacement_um']:.1f}</div><div class='label'>原始平均位移 (µm)</div></div>",
        f"      <div class='metric'><div class='value'>{overall_metrics['rigid_displacement_um']:.1f}</div><div class='label'>刚性配准后平均位移 (µm)</div></div>",
        f"      <div class='metric'><div class='value'>{overall_metrics['non_rigid_displacement_um']:.1f}</div><div class='label'>非刚性配准后平均位移 (µm)</div></div>",
        f"      <div class='metric'><div class='value'>{overall_metrics['rtre']:.2f}%</div><div class='label'>平均相对目标配准误差 (rTRE)</div></div>",
        "    </div>",
        "    <p class='small'>所有距离单位均为微米 (µm)。rTRE 基于共同配准画布对角线计算。</p>",
        "  </div>",
        "  <div class='card'>",
        "    <h2>全片叠加对比（绿色=HE，品红色=IHC）</h2>",
        "    <div class='overlap-grid'>",
    ]

    for entry in overlay_entries:
        title = html.escape(entry["title"])
        slide_parts.append("      <div class='overlap-item'>")
        slide_parts.append(f'        <img src="{entry["data_uri"]}" alt="{title}">')
        slide_parts.append(f'        <div class="caption">{title}</div>')
        slide_parts.append("      </div>")

    slide_parts.extend(
        [
            "    </div>",
            "  </div>",
            "  <div class='card'>",
            "    <h2>逐 Marker 配准结果</h2>",
            "    <table>",
            "      <tr><th>切片</th><th>对齐到</th><th>原始位移 (µm)</th><th>刚性后 (µm)</th><th>非刚性后 (µm)</th><th>rTRE</th><th>匹配点数</th><th>HE 缩略图</th><th>Registered IHC 缩略图</th><th>形变场</th></tr>",
            f'      <tr class=\'he-row\'><td>HE</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td><img class="slide-thumb" src="{he_ref_thumb_uri}" alt="HE reference"></td><td>—</td><td>—</td></tr>',
        ]
    )

    for row in marker_rows:
        rtre_class = "good" if row["rtre"] < rtre_threshold else ""
        slide_parts.append("      <tr>")
        slide_parts.append(f"        <td>{html.escape(row['marker'])}</td>")
        slide_parts.append("        <td>HE</td>")
        slide_parts.append(f"        <td>{row['original_displacement_um']:.1f}</td>")
        slide_parts.append(f"        <td>{row['rigid_displacement_um']:.1f}</td>")
        slide_parts.append(f"        <td>{row['non_rigid_displacement_um']:.1f}</td>")
        slide_parts.append(f'        <td class="{rtre_class}">{row["rtre"]:.2f}%</td>')
        slide_parts.append(f"        <td>{row['n_matches']}</td>")
        slide_parts.append(
            f'        <td><img class="slide-thumb" src="{row["he_thumb_uri"]}" alt="{html.escape(row["marker"])} HE thumbnail"></td>'
        )
        slide_parts.append(
            f'        <td><img class="slide-thumb" src="{row["ihc_thumb_uri"]}" alt="{html.escape(row["marker"])} IHC thumbnail"></td>'
        )
        slide_parts.append(
            f'        <td><img class="slide-thumb" src="{row["def_uri"]}" alt="{html.escape(row["marker"])} deformation field"></td>'
        )
        slide_parts.append("      </tr>")

    slide_parts.extend(
        [
            "    </table>",
            "  </div>",
            "  <div class='card small'>",
            f"    <p>输出路径：{html.escape(str(report_path))}</p>",
            f"    <p>生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            "  </div>",
        ]
    )

    if unified:
        n_patches = len(gallery_entries)
        html_parts.append("  <div class='tab-nav'>")
        html_parts.append(
            "    <button class='active' onclick=\"showTab('slide')\">Slide Summary</button>"
        )
        html_parts.append(
            f"    <button onclick=\"showTab('gallery')\">Patch Gallery ({n_patches})</button>"
        )
        html_parts.append("  </div>")
        html_parts.append("  <div id='panel-slide' class='tab-panel active'>")
        html_parts.extend(slide_parts)
        html_parts.append("  </div>")
        html_parts.append("  <div id='panel-gallery' class='tab-panel'>")
        html_parts.append("    <div class='card gallery-card'>")
        html_parts.append("      <h2>Patch Gallery</h2>")
        if gallery_entries:
            html_parts.append("      <div class='gallery-grid'>")
            for entry in gallery_entries:
                title = html.escape(entry["title"])
                html_parts.append("        <div class='gallery-item'>")
                html_parts.append(
                    f'          <img src="{entry["data_uri"]}" alt="{title}" loading="lazy">'
                )
                html_parts.append(f"          <h4>{title}</h4>")
                html_parts.append("        </div>")
            html_parts.append("      </div>")
        else:
            html_parts.append("      <p class='small'>未采样 patch（viz_sample_n 为 0 或无可视化 patch）。</p>")
        html_parts.append("    </div>")
        html_parts.append("  </div>")
        html_parts.extend(
            [
                "  <script>",
                "    function showTab(name) {",
                "      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));",
                "      document.querySelectorAll('.tab-nav button').forEach(b => b.classList.remove('active'));",
                "      document.getElementById('panel-' + name).classList.add('active');",
                "      event.target.classList.add('active');",
                "    }",
                "  </script>",
            ]
        )
    else:
        html_parts.extend(slide_parts)

    html_parts.extend(["</body>", "</html>"])

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    return output_path

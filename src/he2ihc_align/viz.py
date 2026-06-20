"""Visualization utilities for HE-to-IHC alignment gallery and report."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from skimage import color as skcolor

from he2ihc_align.registration import warp_tools
from he2ihc_align.slide_io.base import Slide

if TYPE_CHECKING:
    from he2ihc_align.registration import non_rigid, rigid


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


def fig_to_data_uri(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG data URI.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure.

    Returns
    -------
    str
        Base64 PNG data URI string.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{data}"


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
        f'  <title>Gallery - {html.escape(slide_id)}</title>',
        "  <style>",
        "    body { font-family: sans-serif; margin: 20px; }",
        "    .entry { margin-bottom: 40px; }",
        "    .entry img { max-width: 100%; border: 1px solid #ccc; }",
        "    .entry h3 { margin-bottom: 5px; }",
        "  </style>",
        "</head>",
        "<body>",
        f'  <h1>Gallery - {html.escape(slide_id)}</h1>',
    ]

    for entry in entries:
        title = html.escape(entry["title"])
        html_parts.append('  <div class="entry">')
        html_parts.append(f'    <h3>{title}</h3>')
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
    from he2ihc_align.registration.rigid import RigidRegistrar

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

    magnitude = np.sqrt(u**2 + v**2)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(magnitude, cmap="viridis", aspect="auto")
    ax.quiver(x, y, u, v, color="white", alpha=0.7, scale=max(w, h) * 2)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def make_marker_thumbnail_figure(
    he_img: np.ndarray,
    ihc_img: np.ndarray,
    non_rigid_registrar: non_rigid.NonRigidRegistrar,
    title: str,
) -> plt.Figure:
    """Create a side-by-side thumbnail: HE and registered IHC."""
    he_gray = _to_gray_uint8(he_img)
    ihc_gray = _to_gray_uint8(ihc_img)
    warped_ihc = non_rigid_registrar.warp_image(ihc_gray, out_shape_rc=he_gray.shape)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(he_gray, cmap="gray")
    axes[0].set_title("HE")
    axes[0].axis("off")

    axes[1].imshow(warped_ihc, cmap="gray")
    axes[1].set_title("Registered IHC")
    axes[1].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def compute_marker_metrics(
    rigid_registrar: rigid.RigidRegistrar,
    non_rigid_registrar: non_rigid.NonRigidRegistrar,
    he_scale: float,
    ihc_scale: float,
) -> dict:
    """Compute per-marker registration metrics in level-0 pixels.

    Returns dict with keys:
    - original_displacement_px
    - rigid_displacement_px
    - non_rigid_displacement_px
    - rtre
    - n_matches
    """
    ref_kp = rigid_registrar.matched_kp_ref
    moving_kp = rigid_registrar.matched_kp_moving

    if ref_kp is None or moving_kp is None or len(ref_kp) == 0:
        return {
            "original_displacement_px": 0.0,
            "rigid_displacement_px": 0.0,
            "non_rigid_displacement_px": 0.0,
            "rtre": 0.0,
            "n_matches": 0,
        }

    ref_kp = np.asarray(ref_kp, dtype=np.float64)
    moving_kp = np.asarray(moving_kp, dtype=np.float64)

    # Original displacement in level-0 pixels
    ref_kp_level0 = ref_kp * he_scale
    moving_kp_level0 = moving_kp * ihc_scale
    original_d = np.mean(warp_tools.calc_d(ref_kp_level0, moving_kp_level0))

    # Rigid displacement: warp HE keypoints to IHC space via inverse rigid
    moving_kp_rigid = rigid_registrar.inverse_warp_xy(ref_kp)
    moving_kp_rigid_level0 = moving_kp_rigid * ihc_scale
    rigid_d = np.mean(warp_tools.calc_d(ref_kp_level0, moving_kp_rigid_level0))

    # Non-rigid displacement: warp HE keypoints to IHC space via full inverse warp
    moving_kp_nr = non_rigid_registrar.inverse_warp_xy(ref_kp)
    moving_kp_nr_level0 = moving_kp_nr * ihc_scale
    nr_d = np.mean(warp_tools.calc_d(ref_kp_level0, moving_kp_nr_level0))

    # rTRE as percentage (relative to image diagonal)
    tre, _ = warp_tools.measure_error(moving_kp_nr, ref_kp, shape_rc=rigid_registrar.ref_img.shape[0:2])
    rtre = tre * 100.0

    return {
        "original_displacement_px": float(original_d),
        "rigid_displacement_px": float(rigid_d),
        "non_rigid_displacement_px": float(nr_d),
        "rtre": float(rtre),
        "n_matches": int(rigid_registrar.n_matches),
    }


def compute_overall_metrics(marker_metrics: dict[str, dict]) -> dict:
    """Aggregate per-marker metrics into overall statistics."""
    if not marker_metrics:
        return {
            "original_displacement_px": 0.0,
            "rigid_displacement_px": 0.0,
            "non_rigid_displacement_px": 0.0,
            "rtre": 0.0,
            "n_matches": 0,
        }

    keys = ["original_displacement_px", "rigid_displacement_px", "non_rigid_displacement_px", "rtre", "n_matches"]
    return {k: float(np.mean([m[k] for m in marker_metrics.values()])) for k in keys}


def create_html_report(
    output_path: Path,
    slide_id: str,
    overall_metrics: dict,
    overlay_entries: list[dict],
    marker_rows: list[dict],
) -> Path:
    """Write a slide-level registration quality report as self-contained HTML.

    Parameters
    ----------
    output_path : Path
        Path to write the HTML file.
    slide_id : str
        Slide/case identifier.
    overall_metrics : dict
        Dict with keys original_displacement_px, rigid_displacement_px,
        non_rigid_displacement_px, rtre, n_matches.
    overlay_entries : list[dict]
        List of dicts with keys ``title`` and ``data_uri``.
    marker_rows : list[dict]
        List of dicts with keys marker, original_displacement_px,
        rigid_displacement_px, non_rigid_displacement_px, rtre, n_matches,
        thumb_uri, def_uri.

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
        f'  <title>Registration Report - {html.escape(slide_id)}</title>',
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2rem; background: #f7f8fa; color: #222; }",
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
        "  </style>",
        "</head>",
        "<body>",
        f'  <h1>🔬 Registration Report – {html.escape(slide_id)}</h1>',
        "  <div class='card'>",
        "    <h2>Overall Error Statistics</h2>",
        "    <div class='metrics'>",
        f"      <div class='metric'><div class='value'>{overall_metrics['original_displacement_px']:.1f}</div><div class='label'>Original mean displacement (px)</div></div>",
        f"      <div class='metric'><div class='value'>{overall_metrics['rigid_displacement_px']:.1f}</div><div class='label'>Rigid mean displacement (px)</div></div>",
        f"      <div class='metric'><div class='value'>{overall_metrics['non_rigid_displacement_px']:.1f}</div><div class='label'>Non-rigid mean displacement (px)</div></div>",
        f"      <div class='metric'><div class='value'>{overall_metrics['rtre']:.2f}%</div><div class='label'>Mean rTRE</div></div>",
        "    </div>",
        "    <p class='small'>All distances are in level-0 pixels. Micrometer support can be added when slide metadata provides MPP.</p>",
        "  </div>",
        "  <div class='card'>",
        "    <h2>Whole-slide Overlay (Green=HE, Magenta=IHC)</h2>",
        "    <div class='overlap-grid'>",
    ]

    for entry in overlay_entries:
        title = html.escape(entry["title"])
        html_parts.append("      <div class='overlap-item'>")
        html_parts.append(f'        <img src="{entry["data_uri"]}" alt="{title}">')
        html_parts.append(f'        <div class="caption">{title}</div>')
        html_parts.append("      </div>")

    html_parts.extend([
        "    </div>",
        "  </div>",
        "  <div class='card'>",
        "    <h2>Per-marker Registration Results</h2>",
        "    <table>",
        "      <tr><th>Marker</th><th>Original (px)</th><th>Rigid (px)</th><th>Non-rigid (px)</th><th>rTRE</th><th>N matches</th><th>Thumbnail</th><th>Deformation</th></tr>",
    ])

    for row in marker_rows:
        rtre_class = "good" if row["rtre"] < 5.0 else ""
        html_parts.append("      <tr>")
        html_parts.append(f'        <td>{html.escape(row["marker"])}</td>')
        html_parts.append(f'        <td>{row["original_displacement_px"]:.1f}</td>')
        html_parts.append(f'        <td>{row["rigid_displacement_px"]:.1f}</td>')
        html_parts.append(f'        <td>{row["non_rigid_displacement_px"]:.1f}</td>')
        html_parts.append(f'        <td class="{rtre_class}">{row["rtre"]:.2f}%</td>')
        html_parts.append(f'        <td>{row["n_matches"]}</td>')
        html_parts.append(f'        <td><img class="slide-thumb" src="{row["thumb_uri"]}" alt="{html.escape(row["marker"])} thumbnail"></td>')
        html_parts.append(f'        <td><img class="slide-thumb" src="{row["def_uri"]}" alt="{html.escape(row["marker"])} deformation field"></td>')
        html_parts.append("      </tr>")

    html_parts.extend([
        "    </table>",
        "  </div>",
        "</body>",
        "</html>",
    ])

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    return output_path

"""Visualization utilities for HE-to-IHC alignment gallery."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from he2ihc_align.slide_io.base import Slide


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


def make_patch_figure(
    he_patch: np.ndarray,
    ihc_patches: dict[str, np.ndarray],
    title: str,
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

    for idx, (marker, patch) in enumerate(ihc_patches.items(), start=1):
        axes[idx].imshow(patch)
        axes[idx].set_title(marker)
        axes[idx].axis("off")

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
        f'  <title>Gallery - {slide_id}</title>',
        "  <style>",
        "    body { font-family: sans-serif; margin: 20px; }",
        "    .entry { margin-bottom: 40px; }",
        "    .entry img { max-width: 100%; border: 1px solid #ccc; }",
        "    .entry h3 { margin-bottom: 5px; }",
        "  </style>",
        "</head>",
        "<body>",
        f'  <h1>Gallery - {slide_id}</h1>',
    ]

    for entry in entries:
        html_parts.append('  <div class="entry">')
        html_parts.append(f'    <h3>{entry["title"]}</h3>')
        html_parts.append(f'    <img src="{entry["data_uri"]}" alt="{entry["title"]}">')
        html_parts.append("  </div>")

    html_parts.extend(["</body>", "</html>"])

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    return output_path

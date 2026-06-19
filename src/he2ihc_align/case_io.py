"""Case discovery utilities."""

from __future__ import annotations

from pathlib import Path


def discover_case(case_dir: str | Path) -> tuple[Path, dict[str, Path]]:
    """Discover HE slide and IHC marker slides within a case directory.

    Args:
        case_dir: Path to the case directory (e.g. test_SCCE/174162-1).

    Returns:
        Tuple of (he_path, markers_dict) where markers_dict maps marker
        name -> Path for each .svs file found.

    Raises:
        FileNotFoundError: If the expected HE .kfb file is not found.
    """
    case_dir = Path(case_dir)
    case_id = case_dir.name

    # Find the HE .kfb whose stem matches the case id
    he_path = None
    for subdir in case_dir.iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.is_file() and f.suffix.lower() == ".kfb" and f.stem == case_id:
                    he_path = f
                    break
        if he_path is not None:
            break

    if he_path is None:
        raise FileNotFoundError(f"HE slide (.kfb matching case id '{case_id}') not found in {case_dir}")

    # Find all .svs files and extract marker names
    markers: dict[str, Path] = {}
    for subdir in case_dir.iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.is_file() and f.suffix.lower() == ".svs":
                    # Marker name is the last space-separated token of the stem
                    marker_name = f.stem.split()[-1]
                    if marker_name in markers:
                        raise ValueError(
                            f"Duplicate marker name '{marker_name}' from {f} and {markers[marker_name]}"
                        )
                    markers[marker_name] = f

    return he_path, markers

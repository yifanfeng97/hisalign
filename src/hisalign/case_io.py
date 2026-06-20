"""Case discovery utilities."""

from __future__ import annotations

from pathlib import Path


def discover_case(case_dir: str | Path) -> tuple[Path, dict[str, Path]]:
    """Discover HE slide and IHC marker slides within a case directory.

    Args:
        case_dir: Path to the case directory (e.g. test_SCCE/174162-1) or
            a batch directory (e.g. test_SCCE/174162-1/174162-1-第一批).

    Returns:
        Tuple of (he_path, markers_dict) where markers_dict maps marker
        name -> Path for each .svs file found.

    Raises:
        FileNotFoundError: If the expected HE .kfb file is not found.
    """
    case_dir = Path(case_dir)
    case_id = case_dir.name

    # First try: look directly in the given directory
    he_path = None
    for f in case_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".kfb" and f.stem == case_id:
            he_path = f
            break

    # If not found, try looking in subdirectories (original behavior)
    if he_path is None:
        for subdir in case_dir.iterdir():
            if subdir.is_dir():
                for f in subdir.iterdir():
                    if f.is_file() and f.suffix.lower() == ".kfb" and f.stem == case_id:
                        he_path = f
                        break
            if he_path is not None:
                break

    if he_path is None:
        # Try with parent name as case_id (for batch directories)
        parent_case_id = case_dir.parent.name
        for f in case_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ".kfb" and f.stem == parent_case_id:
                he_path = f
                break
        if he_path is None:
            raise FileNotFoundError(
                f"HE slide (.kfb matching case id) not found in {case_dir}"
            )
        case_id = parent_case_id

    # Find all .svs files and extract marker names
    # Search both directly in case_dir and in subdirectories
    markers: dict[str, Path] = {}
    search_paths = [case_dir]
    if case_dir != case_dir.parent:
        search_paths.extend([d for d in case_dir.iterdir() if d.is_dir()])

    for search_dir in search_paths:
        for f in search_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ".svs":
                marker_name = f.stem.split()[-1]
                if marker_name in markers:
                    raise ValueError(
                        f"Duplicate marker name '{marker_name}' from {f} and {markers[marker_name]}"
                    )
                markers[marker_name] = f

    return he_path, markers

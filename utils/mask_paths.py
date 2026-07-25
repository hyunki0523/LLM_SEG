from pathlib import Path
from typing import Iterable, Optional, Union


MASK_SUFFIXES = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")


def is_mask_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in MASK_SUFFIXES)


PathLike = Union[str, Path]


def select_preferred_mask_path(path: Optional[PathLike]) -> Optional[Path]:
    """
    Resolve a mask file path. If path is a directory, prefer files with
    'final' in the filename, then fall back to any supported mask file.
    """
    if path is None:
        return None

    path = Path(path)
    if path.is_file():
        return path
    if not path.exists() or not path.is_dir():
        return None

    candidates = [p for p in path.rglob("*") if p.is_file() and is_mask_file(p)]
    if not candidates:
        return None

    final_candidates = [p for p in candidates if "final" in p.name.lower()]
    pool = final_candidates if final_candidates else candidates

    # Prefer shallower paths, then deterministic lexical ordering.
    return sorted(pool, key=lambda p: (len(p.relative_to(path).parts), str(p).lower()))[0]


def find_case_mask_path(
    case_id: str,
    explicit_path: Optional[PathLike] = None,
    search_dirs: Optional[Iterable[PathLike]] = None,
) -> Optional[Path]:
    selected = select_preferred_mask_path(explicit_path)
    if selected is not None:
        return selected

    if search_dirs is None:
        return None

    case_id = str(case_id).strip()
    for search_dir in search_dirs:
        search_dir = Path(search_dir)
        for candidate in (
            search_dir / f"{case_id}.nii.gz",
            search_dir / f"{case_id}.nii",
            search_dir / case_id,
            search_dir / case_id / "final.nii.gz",
            search_dir / case_id / "mask_final.nii.gz",
            search_dir / case_id / "mask.nii.gz",
            search_dir / case_id / "mask.nii",
        ):
            selected = select_preferred_mask_path(candidate)
            if selected is not None:
                return selected

    return None

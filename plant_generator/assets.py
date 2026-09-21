"""Locate the paired fruit and calyx meshes used by the plant builder."""

from pathlib import Path
import re


def paired_leaf_mesh(berry_path: Path) -> Path:
    if berry_path.stem.lower() == "berry":
        leaf_path = berry_path.with_name("leaves_0001.obj")
    else:
        match = re.fullmatch(r"strawberry_(\d+)", berry_path.stem, re.IGNORECASE)
        if match is None:
            raise ValueError(f"Cannot match a leaf mesh to {berry_path}")
        leaf_name = f"leaves_{match.group(1)}.obj"
        leaf_path = (berry_path.parent.parent / "leaves" / leaf_name
                     if berry_path.parent.name.lower() == "berries"
                     else berry_path.with_name(leaf_name))
    if not leaf_path.is_file():
        raise FileNotFoundError(
            f"Missing leaf mesh for {berry_path}: expected {leaf_path}. "
            "Generate paired meshes before launching the plant."
        )
    return leaf_path


def find_berry_mesh_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    berry_meshes = sorted(
        path for path in dataset_dir.rglob("*.obj")
        if path.stem.lower().startswith(("berry", "strawberry_"))
    )
    if not berry_meshes:
        raise FileNotFoundError(
            f"No berry OBJ meshes in {dataset_dir}. "
            "Run view_strawberry_mujoco.py --check-only to populate the dataset."
        )
    return [(path, paired_leaf_mesh(path)) for path in berry_meshes]

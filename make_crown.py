"""MuJoCo strawberry plant generator.

Run:  python make_crown.py
"""
from plant_generator.params import PlantParams
from plant_generator.mjcf_builder import build_xml
from plant_generator.geometry import exp_stiffness, logistic_stiffness, taper_radius, box_mass, ellipsoid_mass
from pathlib import Path
import re

import mujoco
import mujoco.viewer


# Set to True to use berry OBJ meshes found under dataset/.
# False keeps the original primitive ellipsoid berries.
USE_DATASET_BERRIES = True
DATASET_DIR = Path(__file__).resolve().parent / "dataset"


def paired_leaf_mesh(berry_path: Path) -> Path:
    """Find the calyx OBJ generated with this berry in either dataset layout."""
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


def main():
    params = PlantParams()           # defaults match current behavior
    berry_mesh_pairs = None
    if USE_DATASET_BERRIES:
        berry_meshes = sorted(
            path for path in DATASET_DIR.rglob("*.obj")
            if path.stem.lower().startswith(("berry", "strawberry_"))
        )
        if not berry_meshes:
            raise FileNotFoundError(
                f"No berry OBJ meshes in {DATASET_DIR}. "
                "Run view_strawberry_mujoco.py --check-only to populate the dataset."
            )
        berry_mesh_pairs = [(path, paired_leaf_mesh(path)) for path in berry_meshes]
        print(f"Using {len(berry_mesh_pairs)} berry and leaf OBJ pair(s) from {DATASET_DIR}")
    xml = build_xml(params, berry_mesh_pairs=berry_mesh_pairs)

    xml_path = Path("tmp/crown_petioles.xml")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    with xml_path.open('w') as f:
        f.write(xml)
    print(f"Wrote {xml_path}  ({len(xml)} bytes)")

    # ── weight report ──
    lb = params.leaf_blade
    leaflet_mass = box_mass(lb.length, lb.width, lb.thickness, lb.density)
    print(f"Leaflet ({lb.length*100:.0f}x{lb.width*100:.0f}x{lb.thickness*1000:.0f} cm, "
          f"density={lb.density:.0f}): {leaflet_mass*1000:.2f} g each, "
          f"{3*leaflet_mass*1000:.2f} g per trifoliate")

    for name, br in [("R=12mm", 0.012), ("R=11mm", 0.011), ("R=10mm", 0.010)]:
        m = ellipsoid_mass(br * 1.15, br, br, 950)
        print(f"Berry {name} (ellipsoid, density=950): {m * 1000:.2f} g")

    # stiffness report
    bp = params.berry_profile
    print(f"\nBerry petiole stiffness gradient (n_seg={bp.n_seg}):")
    for i in range(bp.n_seg):
        k = bp.stiffness(i, bp.n_seg) * 2  # report undivided stiffness
        print(f"  seg{i}: k={k:.4f} N·m/rad, r={taper_radius(bp.r_base, bp.taper, i, bp.n_seg) * 1000:.2f} mm")

    lp = params.leaf_profile
    print(f"\nLeaf petiole stiffness gradient (n_seg={lp.n_seg}):")
    for i in range(lp.n_seg):
        k = lp.stiffness(i, lp.n_seg) * 2
        print(f"  seg{i}: k={k:.4f} N·m/rad, r={taper_radius(lp.r_base, lp.taper, i, lp.n_seg) * 1000:.2f} mm")

    # Find first truss to report stem stiffness
    for leaf in params.leaves:
        if leaf.truss is not None:
            tc = leaf.truss
            print(f"\nTruss stem stiffness gradient (n_seg={tc.stem_n_seg}):")
            for i in range(tc.stem_n_seg):
                k = exp_stiffness(tc.stem_k_base, tc.stem_k_decay, i, tc.stem_n_seg)
                print(f"  seg{i}: k={k:.4f} N·m/rad, "
                      f"r={taper_radius(tc.stem_r_base, tc.stem_taper, i, tc.stem_n_seg) * 1000:.2f} mm")
            break

    print("\nLaunching viewer…")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.viewer.launch(model, data)
    print("Viewer closed.")


if __name__ == "__main__":
    main()

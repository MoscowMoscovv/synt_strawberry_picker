"""Generate N varied strawberry plants beside the original one in one MJCF scene.

Example: python generate_plant_grid.py --count 8 --seed 42 --view
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plant_generator.assets import find_berry_mesh_pairs
from plant_generator.mjcf_builder import build_grid_xml
from plant_generator.procedural import sample_plants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True,
                        help="number of generated plants in addition to the original")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for reproducible plant variation")
    parser.add_argument("--spacing", type=float, default=0.7,
                        help="grid cell spacing in metres (default: 0.7)")
    parser.add_argument("--output", type=Path, default=Path("tmp/plant_grid.xml"),
                        help="MJCF file to write")
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path(__file__).resolve().parent / "dataset",
                        help="directory containing paired berry and leaf OBJ meshes")
    parser.add_argument("--primitive-berries", action="store_true",
                        help="use ellipsoid berries instead of paired dataset OBJ meshes")
    parser.add_argument("--view", action="store_true", help="open the MuJoCo viewer")
    args = parser.parse_args()

    if args.count < 0:
        parser.error("--count must be nonnegative")
    pairs = None
    if not args.primitive_berries:
        pairs = find_berry_mesh_pairs(args.dataset_dir)
        print(f"Using {len(pairs)} berry and calyx mesh pair(s)")
    xml = build_grid_xml(sample_plants(args.count, args.seed), args.spacing, pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(xml, encoding="utf-8")
    print(f"Wrote {args.output} with 1 original + {args.count} generated plant(s)")

    if args.view:
        import mujoco
        import mujoco.viewer

        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()

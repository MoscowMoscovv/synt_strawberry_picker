"""Generate or reuse one strawberry and leaf meshes, then inspect them in MuJoCo.

Run with the project's Python environment:
    .venv/Scripts/python view_strawberry_mujoco.py --leaf-count 1

Use --check-only to generate assets and validate the MuJoCo model without a GUI.
No physics steps are run; all mesh geoms are attached to the world body.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from batch_generate_strawberries import build_jobs, parse_args as generator_args


ROOT = Path(__file__).resolve().parent
GENERATOR = ROOT / "batch_generate_strawberries.py"
DATASET = ROOT / "dataset"
STEAM_BLENDER = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf-count", type=int, default=1, help="number of leaf meshes to import")
    parser.add_argument("--seed", type=int, default=42, help="reproducible geometry seed")
    parser.add_argument("--subdivide", type=int, default=5, help="berry subdivision level (0-10)")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument("--blender", default=os.environ.get("BLENDER_EXE", str(STEAM_BLENDER)))
    parser.add_argument("--refresh", action="store_true", help="regenerate and replace this preview's meshes")
    parser.add_argument("--check-only", action="store_true", help="compile scene without opening a viewer")
    args = parser.parse_args()
    if args.leaf_count < 1:
        parser.error("--leaf-count must be at least 1")
    if not 0 <= args.subdivide <= 10:
        parser.error("--subdivide must be between 0 and 10")
    return args


def planned_meshes(seed: int, subdivide: int, leaf_count: int) -> list[dict]:
    pair = build_jobs(generator_args(["--count", "1", "--seed", str(seed),
                                      "--subdivide", str(subdivide)]))
    berry_params = pair[0]["parameters"]
    first_leaf_params = pair[1]["parameters"]
    shared = {key: value for key, value in first_leaf_params.items()
              if key != "Leaves/Scale leaves"}
    extra_rng = random.Random(seed ^ 0x5EED)
    meshes = [{"name": "berry", "part": "all", "parameters": berry_params,
               "file": "berry.obj"}]
    for index in range(1, leaf_count + 1):
        scale = (first_leaf_params["Leaves/Scale leaves"] if index == 1
                 else round(extra_rng.uniform(1.2, 1.8), 6))
        meshes.append({"name": f"leaves_{index:04d}", "part": "leaves",
                       "parameters": {**shared, "Leaves/Scale leaves": scale},
                       "file": f"leaves_{index:04d}.obj"})
    return meshes


def ensure_meshes(args: argparse.Namespace, preview_dir: Path, plan: list[dict]) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    missing = [entry for entry in plan if args.refresh or not (preview_dir / entry["file"]).is_file()]
    if missing:
        print(f"Generating {len(missing)} missing mesh(es) with Blender...", flush=True)
        with tempfile.TemporaryDirectory(prefix="build_", dir=preview_dir) as temporary:
            temp = Path(temporary)
            config = temp / "variants.json"
            config.write_text(json.dumps({"variants": [
                {"name": entry["name"], "part": entry["part"],
                 "parameters": entry["parameters"]} for entry in missing
            ]}, indent=2), encoding="utf-8")
            output = temp / "exports"
            subprocess.run([sys.executable, str(GENERATOR), "--blender", args.blender,
                            "--config", str(config), "--output-dir", str(output)], check=True)
            for entry in missing:
                source = output / entry["file"]
                if not source.is_file() or source.stat().st_size == 0:
                    raise RuntimeError(f"Blender did not export {entry['file']}")
                shutil.copy2(source, preview_dir / entry["file"])
                material = source.with_suffix(".mtl")
                if material.is_file():
                    shutil.copy2(material, (preview_dir / entry["file"]).with_suffix(".mtl"))
    else:
        print(f"Reusing {len(plan)} mesh(es) from {preview_dir}", flush=True)

    manifest = {"seed": args.seed, "subdivide": args.subdivide,
                "source_blend": str(ROOT / "Strawberry" / "Strawberry" / "Procedual_strawberry.blend"),
                "meshes": plan}
    (preview_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


def build_xml(plan: list[dict], preview_dir: Path) -> Path:
    root = ET.Element("mujoco", model="strawberry_import_preview")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "statistic", center="0 0 0.04", extent="0.15")
    assets = ET.SubElement(root, "asset")
    for entry in plan:
        ET.SubElement(assets, "mesh", name=entry["name"], file=entry["file"])

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="key", pos="-0.15 -0.12 0.25",
                  dir="0.4 0.5 -0.8", directional="true")
    ET.SubElement(world, "camera", name="inspection", pos="0 -0.18 0.09",
                  xyaxes="1 0 0 0 0.267 0.964", fovy="35")
    ET.SubElement(world, "geom", name="berry_visual", type="mesh", mesh="berry",
                  rgba="0.85 0.07 0.12 1", contype="0", conaffinity="0")
    leaves = plan[1:]
    for index, entry in enumerate(leaves):
        angle = 2 * math.pi * index / len(leaves)
        quat = f"{math.cos(angle / 2):.9g} 0 0 {math.sin(angle / 2):.9g}"
        ET.SubElement(world, "geom", name=f"leaves_visual_{index + 1:04d}",
                      type="mesh", mesh=entry["name"], quat=quat,
                      rgba="0.14 0.42 0.10 1", contype="0", conaffinity="0")
    xml_path = preview_dir / "preview.xml"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(xml_path, encoding="unicode", xml_declaration=True)
    return xml_path


def main() -> None:
    args = parse_args()
    import mujoco

    preview_dir = args.dataset_dir.resolve() / f"seed_{args.seed}_subdiv_{args.subdivide}"
    plan = planned_meshes(args.seed, args.subdivide, args.leaf_count)
    ensure_meshes(args, preview_dir, plan)
    xml_path = build_xml(plan, preview_dir)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if model.nmesh != len(plan) or model.ngeom != len(plan):
        raise RuntimeError(f"Expected {len(plan)} meshes and geoms, got {model.nmesh} and {model.ngeom}")
    print(f"MuJoCo imported {model.nmesh} meshes and {model.ngeom} static geoms: {xml_path}", flush=True)
    if args.check_only:
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.camera("inspection").id
        viewer.sync()
        print("Static viewer open. Close the MuJoCo window to exit.", flush=True)
        while viewer.is_running():
            # No mj_step: this is only an import and appearance inspection.
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()

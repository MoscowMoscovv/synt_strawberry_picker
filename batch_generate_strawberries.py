"""Generate paired strawberry-body and leaf OBJ meshes with Blender.

Run with ordinary Python; this file relaunches itself inside Blender 4.3:

    python batch_generate_strawberries.py --count 10 --seed 42 --output-dir output

Alternatively, --config accepts a JSON ``variants`` list. Each variant has a
unique ``name``, optional ``part`` ("all" or "leaves"), and ``parameters`` keyed
by interface label (e.g. ``Noise shape/W``) or socket ID (e.g. ``Socket_12``).
Omitted parameters retain the source object's saved modifier values.
"""

import argparse
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
DEFAULT_BLEND = HERE / "Strawberry" / "Strawberry" / "Procedual_strawberry.blend"
SOURCE_OBJECT = "Strawberry_procedual"  # spelling in the actual .blend file


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, default=DEFAULT_BLEND)
    parser.add_argument("--blender", default=os.environ.get("BLENDER_EXE", "blender"))
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated_strawberries")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--count", type=int, metavar="N", help="generate N matched berry/leaf pairs")
    mode.add_argument("--config", type=Path, help="export individually specified variants")
    parser.add_argument("--seed", type=int, default=0, help="reproducible batch random seed")
    parser.add_argument("--scale-range", type=float, nargs=2, default=(0.9, 1.15), metavar=("MIN", "MAX"))
    parser.add_argument("--w-range", type=float, nargs=2, default=(0.0, 100.0), metavar=("MIN", "MAX"))
    parser.add_argument("--leaf-scale-range", type=float, nargs=2, default=(1.2, 1.8), metavar=("MIN", "MAX"))
    parser.add_argument("--subdivide", type=int, help="override berry subdivision level (0-10)")
    parser.add_argument("--max-faces", type=int,
                        help="simplify each exported mesh to at most this many triangles (>= 4)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing generated files")
    parser.add_argument("--list-parameters", action="store_true")
    args = parser.parse_args(argv)
    if args.max_faces is not None and args.max_faces < 4:
        parser.error("--max-faces must be at least 4")
    return args


def load_variants(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    variants = data["variants"]
    if not isinstance(variants, list) or not variants:
        raise ValueError("config.variants must be a nonempty list")
    names = set()
    for item in variants:
        name = item["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError(f"Invalid or duplicate variant name: {name!r}")
        names.add(name)
        if not isinstance(item.get("parameters", {}), dict):
            raise ValueError(f"{name}: parameters must be an object")
        if item.get("part", "all") not in ("all", "leaves"):
            raise ValueError(f"{name}: part must be 'all' or 'leaves'")
    return variants


def build_jobs(args):
    if args.config:
        return [{**item, "directory": ""} for item in load_variants(args.config)]
    if args.count is None or args.count < 1:
        raise ValueError("Specify --count N (N >= 1), --config, or --list-parameters")
    if args.subdivide is not None and not 0 <= args.subdivide <= 10:
        raise ValueError("--subdivide must be between 0 and 10")
    for label, (low, high) in (("scale", args.scale_range), ("W", args.w_range),
                               ("leaf scale", args.leaf_scale_range)):
        if low <= 0 and label != "W":
            raise ValueError(f"{label} range must be positive")
        if low > high:
            raise ValueError(f"{label} range minimum exceeds maximum")
    rng = random.Random(args.seed)
    jobs = []
    for index in range(1, args.count + 1):
        shared = {
            "Divots/Seed": rng.randrange(0, 2**31 - 1),
            "Noise shape/W": round(rng.uniform(*args.w_range), 6),
            "Main/Scale all": round(rng.uniform(*args.scale_range), 6),
        }
        if args.subdivide is not None:
            shared["Main/Subdivide"] = args.subdivide
        leaf_scale = round(rng.uniform(*args.leaf_scale_range), 6)
        suffix = f"{index:04d}"
        jobs.append({"name": f"strawberry_{suffix}", "directory": "berries",
                     "part": "all", "parameters": {**shared, "Leaves/On\\off leaves": False}})
        jobs.append({"name": f"leaves_{suffix}", "directory": "leaves",
                     "part": "leaves", "parameters": {**shared, "Leaves/Scale leaves": leaf_scale}})
    return jobs


def external_main(args):
    if not args.blend.is_file():
        raise FileNotFoundError(args.blend)
    if not args.list_parameters:
        build_jobs(args)  # catch input errors before starting Blender
    command = [str(args.blender), "--background", str(args.blend.resolve()),
               "--python-exit-code", "1", "--python", str(Path(__file__).resolve()),
               "--", "--blend", str(args.blend.resolve()), "--output-dir",
               str(args.output_dir.resolve())]
    if args.count is not None:
        command.extend(["--count", str(args.count)])
    if args.config:
        command.extend(["--config", str(args.config.resolve())])
    command.extend(["--seed", str(args.seed)])
    for flag, pair in (("--scale-range", args.scale_range), ("--w-range", args.w_range),
                       ("--leaf-scale-range", args.leaf_scale_range)):
        command.extend([flag, str(pair[0]), str(pair[1])])
    if args.subdivide is not None:
        command.extend(["--subdivide", str(args.subdivide)])
    if args.max_faces is not None:
        command.extend(["--max-faces", str(args.max_faces)])
    if args.overwrite:
        command.append("--overwrite")
    if args.list_parameters:
        command.append("--list-parameters")
    subprocess.run(command, check=True)


def input_sockets(group):
    for item in group.interface.items_tree:
        if item.item_type == "SOCKET" and item.in_out == "INPUT" and item.socket_type != "NodeSocketGeometry":
            panel = item.parent.name if item.parent and item.parent.name else "Main"
            yield item, f"{panel}/{item.name}"


def set_parameters(modifier, parameters):
    entries = list(input_sockets(modifier.node_group))
    by_path = {path: item for item, path in entries}
    by_id = {item.identifier: item for item, _ in entries}
    by_name = {}
    for item, _ in entries:
        by_name.setdefault(item.name, []).append(item)
    for key, value in parameters.items():
        item = by_path.get(key) or by_id.get(key)
        if item is None and len(by_name.get(key, [])) == 1:
            item = by_name[key][0]
        if item is None:
            raise ValueError(f"Unknown or ambiguous input {key!r}; use panel/name or socket ID")
        if item.socket_type == "NodeSocketBool":
            if type(value) is not bool:
                raise TypeError(f"{key}: expected boolean")
        elif item.socket_type == "NodeSocketInt":
            if type(value) is not int:
                raise TypeError(f"{key}: expected integer")
        elif item.socket_type == "NodeSocketFloat":
            if type(value) not in (int, float):
                raise TypeError(f"{key}: expected number")
            value = float(value)
        else:
            raise TypeError(f"{key}: only numeric and boolean inputs can be set via JSON")
        modifier[item.identifier] = value
        print(f"  {key} [{item.identifier}] = {modifier[item.identifier]!r}", flush=True)


def route_leaves_only(modifier):
    """Use this asset's leaf branch as the node group's geometry output."""
    group = modifier.node_group.copy()  # never modify the source object's tree
    modifier.node_group = group
    leaf_links = [link for link in group.links
                  if link.from_node.type == "GROUP_INPUT"
                  and link.from_socket.name == "On\\off leaves"
                  and link.to_node.bl_idname == "GeometryNodeSwitch"]
    if len(leaf_links) != 1:
        raise RuntimeError("Could not identify the procedural leaf switch")
    leaf_switch = leaf_links[0].to_node
    material_links = [link for link in group.links
                      if link.to_node.type == "GROUP_OUTPUT"
                      and link.from_node.bl_idname == "GeometryNodeSetMaterial"]
    if len(material_links) != 1:
        raise RuntimeError("Could not identify the final material/output node")
    material_node = material_links[0].from_node
    leaf_input = next(item for item, _ in input_sockets(group)
                      if item.name == "On\\off leaves")
    modifier[leaf_input.identifier] = True
    group.links.new(leaf_switch.outputs["Output"], material_node.inputs["Geometry"])


def blender_main(args):
    import bpy

    original = bpy.data.objects.get(SOURCE_OBJECT)
    if original is None or original.type != "MESH":
        raise RuntimeError(f"Missing mesh object {SOURCE_OBJECT!r}")
    source_modifier = next((m for m in original.modifiers if m.type == "NODES"), None)
    if source_modifier is None or source_modifier.node_group is None:
        raise RuntimeError(f"{SOURCE_OBJECT}: missing Geometry Nodes modifier")

    if args.list_parameters:
        for item, path in input_sockets(source_modifier.node_group):
            actual = source_modifier.get(item.identifier, item.default_value)
            label = getattr(actual, "name", actual)
            print(f"{path:32} {item.identifier:10} current={label}", flush=True)
        return

    jobs = build_jobs(args)
    destinations = [args.output_dir / job["directory"] / f"{job['name']}.obj" for job in jobs]
    if args.count is not None and args.output_dir.exists():
        expected = {path.resolve() for path in destinations}
        extra_objs = [str(path) for path in args.output_dir.rglob("*.obj")
                      if path.resolve() not in expected]
        if extra_objs and not args.overwrite:
            raise FileExistsError("Output directory contains OBJ files outside this batch; "
                                  "choose an empty --output-dir: " + ", ".join(extra_objs[:3]))
    if not args.overwrite:
        occupied = [str(path) for path in destinations if path.exists() or path.with_suffix(".mtl").exists()]
        if (args.output_dir / "manifest.json").exists():
            occupied.append(str(args.output_dir / "manifest.json"))
        if occupied:
            raise FileExistsError("Output already exists; choose another --output-dir or use --overwrite: "
                                  + ", ".join(occupied[:3]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    records = []

    for job, dest in zip(jobs, destinations):
        variant = job
        name = variant["name"]
        ob = original.copy()
        ob.data = original.data.copy()
        ob.name = f"generated_{name}"
        bpy.context.scene.collection.objects.link(ob)
        mod = next(m for m in ob.modifiers if m.type == "NODES")
        print(f"[{name}] applying parameters", flush=True)
        set_parameters(mod, variant.get("parameters", {}))
        if variant.get("part", "all") == "leaves":
            route_leaves_only(mod)
        ob.update_tag()
        depsgraph.update()

        ev = ob.evaluated_get(depsgraph)
        mesh = ev.to_mesh()
        mesh.calc_loop_triangles()
        original_triangles = len(mesh.loop_triangles)
        ev.to_mesh_clear()
        if args.max_faces is not None and original_triangles > args.max_faces:
            # Put simplification after Geometry Nodes, and count triangles rather
            # than OBJ polygons: MuJoCo triangulates quads during loading.
            ob.modifiers.new(name="Export triangulation", type="TRIANGULATE")
            decimate = ob.modifiers.new(name="Export face limit", type="DECIMATE")
            decimate.ratio = args.max_faces / original_triangles
            decimate.use_collapse_triangulate = True
            ob.update_tag()
            depsgraph.update()

        ev = ob.evaluated_get(depsgraph)
        mesh = ev.to_mesh()
        mesh.calc_loop_triangles()
        vertices, faces = len(mesh.vertices), len(mesh.polygons)
        triangles = len(mesh.loop_triangles)
        ev.to_mesh_clear()
        if args.max_faces is not None and triangles > args.max_faces:
            raise RuntimeError(f"{name}: simplification produced {triangles} triangles; "
                               f"requested at most {args.max_faces}")
        if faces == 0 or (variant.get("part", "all") != "leaves" and faces <= len(original.data.polygons)):
            raise RuntimeError(f"{name}: evaluated only {faces} faces; node tree may not have run")

        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        dest.parent.mkdir(parents=True, exist_ok=True)
        result = bpy.ops.wm.obj_export(
            filepath=str(dest), export_selected_objects=True,
            apply_modifiers=True, export_eval_mode="DAG_EVAL_VIEWPORT",
            forward_axis="Y", up_axis="Z", export_materials=True,
        )
        if "FINISHED" not in result or not dest.is_file():
            raise RuntimeError(f"OBJ export failed for {name}")
        print(f"[{name}] {vertices} vertices, {faces} faces -> {dest}", flush=True)
        records.append({"file": str(dest.relative_to(args.output_dir)).replace("\\", "/"),
                        "part": variant.get("part", "all"), "parameters": variant.get("parameters", {}),
                        "vertices": vertices, "faces": faces, "triangles": triangles})
        ob.select_set(False)
        copied_mesh = ob.data
        copied_group = mod.node_group if mod.node_group != source_modifier.node_group else None
        bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.meshes.remove(copied_mesh)
        if copied_group:
            bpy.data.node_groups.remove(copied_group)

    manifest = {"source_blend": str(args.blend), "source_object": SOURCE_OBJECT,
                "seed": args.seed if args.count is not None else None,
                "pairs": args.count, "max_faces": args.max_faces, "meshes": records}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    # Blender puts arguments after a standalone '--' into sys.argv.
    in_blender = "bpy" in sys.modules
    argv = sys.argv[sys.argv.index("--") + 1:] if in_blender and "--" in sys.argv else sys.argv[1:]
    parsed = parse_args(argv)
    if in_blender:
        blender_main(parsed)
    else:
        external_main(parsed)

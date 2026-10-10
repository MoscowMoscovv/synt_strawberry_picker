"""Create a MuJoCo scene containing one procedural strawberry plant and an AUBO i5.

The AUBO i5 model is taken from Manipulator-Mujoco::

    git clone https://github.com/ian-chuang/Manipulator-Mujoco.git

    python create_picker_scene.py \
        --aubo-dir Manipulator-Mujoco/manipulator_mujoco/assets/robots/aubo_i5 \
        --primitive-berries --validate --view

The complete AUBO i5 directory is copied next to the generated XML. This keeps
the robot include and mesh paths valid when the output scene is moved.
Dataset meshes are bundled in the same way when ``--bundle-assets`` is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from plant_generator.assets import find_berry_mesh_pairs
from plant_generator.mjcf_builder import build_xml
from plant_generator.procedural import sample_plants


def _vec3(value: str, option: str) -> tuple[float, float, float]:
    """Parse a MuJoCo-style three-value position."""
    try:
        values = tuple(float(part) for part in value.replace(",", " ").split())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option} must contain three numbers") from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"{option} must contain three numbers")
    return values  # type: ignore[return-value]


def _copy_robot_package(source: Path, output: Path, gripper_source: Path) -> Path:
    """Copy the AUBO and AG95 packages beside output and return their directory."""
    destination = output.parent / f"{output.stem}_aubo_i5"
    source = source.resolve()
    destination = destination.resolve()
    if source != destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
    if not gripper_source.is_dir():
        raise FileNotFoundError(f"AG95 gripper model directory not found: {gripper_source}")
    shutil.copytree(gripper_source, destination / "ag95", dirs_exist_ok=True)
    repo_license = source.parents[3] / "LICENSE"
    if repo_license.is_file():
        shutil.copy2(repo_license, destination / "LICENSE")
    return Path(destination.name)


def _prepare_aubo_model(package: Path, position: tuple[float, float, float]) -> None:
    """Attach AG95, resolve both mesh paths, and add arm actuators."""
    model = package / "aubo_i5.xml"
    if not model.is_file():
        return
    root = ET.parse(model).getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        # MuJoCo applies compiler meshdir from the top-level document when
        # resolving assets from an included MJCF. Make paths explicit so the
        # model also works when this file is included from the composed scene.
        meshdir = compiler.get("meshdir")
        if meshdir:
            for mesh in root.findall("./asset/mesh"):
                filename = mesh.get("file")
                if filename and not Path(filename).is_absolute():
                    mesh.set("file", (Path(meshdir) / filename).as_posix())
            compiler.attrib.pop("meshdir", None)
        texturedir = compiler.get("texturedir")
        if texturedir:
            for texture in root.findall("./asset/texture"):
                filename = texture.get("file")
                if filename and not Path(filename).is_absolute():
                    texture.set("file", (Path(texturedir) / filename).as_posix())
            compiler.attrib.pop("texturedir", None)
    world = root.find("worldbody")
    if world is None:
        return
    body = next((candidate for candidate in world.findall("body")
                 if candidate.get("name", "").lower() == "base_link"), None)
    if body is not None:
        _set_body_position(body, position)

    # Merge the AG95 MJCF into the arm MJCF and mount its base at the arm's
    # attachment site, matching the source repository's AuboI5Env transform.
    gripper_path = package / "ag95" / "ag95.xml"
    gripper = ET.parse(gripper_path).getroot()
    gripper_compiler = gripper.find("compiler")
    meshdir = gripper_compiler.get("meshdir", "") if gripper_compiler is not None else ""
    for mesh in gripper.findall("./asset/mesh"):
        filename = mesh.get("file")
        if filename and not Path(filename).is_absolute():
            mesh.set("file", (Path("ag95") / meshdir / filename).as_posix())

    # Keep AG95 defaults isolated by giving its visual/collision classes
    # distinct names. Its joint and general-actuator defaults fill defaults
    # that are absent from the AUBO model.
    class_names = {
        "visual": "ag95_visual",
        "collision": "ag95_collision",
        "pad_box_0": "ag95_pad_box_0",
        "pad_box_1": "ag95_pad_box_1",
        "sphere_collision": "ag95_sphere_collision",
    }
    for element in gripper.iter():
        for attribute in ("class", "childclass"):
            if element.get(attribute) in class_names:
                element.set(attribute, class_names[element.get(attribute)])
        # The stock fingertip contacts are very small and use MuJoCo's low
        # default friction. A thin plant stem can therefore be pushed along
        # the pad and spring out before the fingers settle. Give only the
        # AG95's contact pads and fingertip proxies a grippy, slightly softer
        # contact; visual meshes and the rest of the robot stay unchanged.
        if element.tag == "geom" and element.get("class") in {
            "ag95_pad_box_0", "ag95_pad_box_1", "ag95_sphere_collision",
        }:
            element.set("friction", "2.0 0.1 0.01")
            element.set("solref", "0.01 1")

    robot_default = root.find("default")
    if robot_default is None:
        robot_default = ET.SubElement(root, "default")
    gripper_default = gripper.find("default")
    if gripper_default is not None:
        for element in list(gripper_default):
            robot_default.append(element)

    root_asset = root.find("asset")
    gripper_asset = gripper.find("asset")
    if root_asset is None:
        root_asset = ET.Element("asset")
        root.insert(list(root).index(robot_default), root_asset)
    if gripper_asset is not None:
        root_asset.extend(list(gripper_asset))

    gripper_world = gripper.find("worldbody")
    mount = next((node for node in world.iter("body")
                  if node.get("name") == "tool0_link"), None) if world is not None else None
    gripper_bodies = list(gripper_world) if gripper_world is not None else []
    if mount is None or not any(body.get("name") == "base" for body in gripper_bodies):
        raise ValueError("AUBO attachment_site/tool0_link or AG95 base body is missing")
    gripper_frame = ET.SubElement(mount, "body", {
        "name": "ag95_mount", "pos": "0 0 0", "quat": "0 0 0 1",
    })
    gripper_frame.extend(gripper_bodies)

    # Bring over the gripper's tendon, coupling constraints and actuator.
    for tag in ("contact", "equality", "tendon", "actuator"):
        source_section = gripper.find(tag)
        if source_section is not None:
            target_section = root.find(tag)
            if target_section is None:
                target_section = ET.SubElement(root, tag)
            target_section.extend(list(source_section))
    gripper_option = gripper.find("option")
    robot_option = root.find("option")
    if gripper_option is not None and robot_option is not None:
        if gripper_option.get("impratio"):
            robot_option.set("impratio", gripper_option.get("impratio"))

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for joint_name in ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"):
        if not any(item.get("joint") == joint_name for item in actuator):
            ET.SubElement(actuator, "position", {
                "name": joint_name.removesuffix("_joint"), "joint": joint_name,
                "kp": "100", "kv": "50", "ctrlrange": "-3.04 3.04",
                "forcerange": "-150 150",
            })
    model.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")


def _load_robot_root(aubo_dir: Path) -> tuple[ET.Element, Path]:
    """Load the AUBO scene file, falling back to its robot MJCF."""
    for filename in ("scene.xml", "aubo_i5.xml"):
        path = aubo_dir / filename
        if path.is_file():
            return ET.parse(path).getroot(), path
    raise FileNotFoundError(
        f"{aubo_dir} does not contain scene.xml or aubo_i5.xml. "
        "Pass --aubo-dir pointing at the Manipulator-Mujoco AUBO i5 model directory."
    )


def _set_body_position(body: ET.Element, position: tuple[float, float, float]) -> None:
    body.set("pos", " ".join(f"{part:.6g}" for part in position))


def _bundle_plant_meshes(root: ET.Element, output: Path, dataset_dir: Path) -> None:
    """Copy plant OBJ/MTL files and replace absolute mesh paths with relative ones."""
    asset_dir = output.parent / f"{output.stem}_assets"
    dataset_dir = dataset_dir.resolve()
    for mesh in root.findall("./asset/mesh"):
        source = Path(mesh.get("file", "")).resolve()
        try:
            relative_source = source.relative_to(dataset_dir)
        except ValueError as exc:
            raise ValueError(f"Plant mesh {source} is outside dataset directory {dataset_dir}") from exc
        destination = asset_dir / relative_source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        material = source.with_suffix(".mtl")
        if material.is_file():
            shutil.copy2(material, destination.with_suffix(".mtl"))
        mesh.set("file", (Path(asset_dir.name) / relative_source).as_posix())


def build_picker_xml(
    aubo_dir: Path,
    *,
    seed: int | None = None,
    plant_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    robot_position: tuple[float, float, float] = (0.55, 0.0, 0.0),
    berry_mesh_pairs: list[tuple[Path, Path]] | None = None,
    copied_robot_dir: Path | None = None,
) -> str:
    """Compose the AUBO i5 root with one sampled plant body.

    ``copied_robot_dir`` is the directory name relative to the eventual XML
    file. It is used to rewrite the robot include after copying.
    """
    root, source_file = _load_robot_root(aubo_dir)
    if root.tag != "mujoco":
        raise ValueError(f"Expected a <mujoco> root in {source_file}")
    root.set("model", "aubo_i5_strawberry_picker")

    if copied_robot_dir is not None:
        for include in root.findall("./include"):
            include_file = include.get("file")
            if include_file:
                include.set("file", (copied_robot_dir / include_file).as_posix())

    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")

    plant_params = sample_plants(1, seed)[0]
    plant_root = ET.fromstring(build_xml(plant_params, berry_mesh_pairs))
    plant_world = plant_root.find("worldbody")
    if plant_world is None:
        raise ValueError("Plant builder did not produce a worldbody")
    plant_body = plant_world.find("body[@name='crown']")
    if plant_body is None:
        raise ValueError("Plant builder did not produce a crown body")
    _set_body_position(plant_body, plant_position)
    # Reuse the robot scene's floor when present; otherwise add the plant floor.
    has_ground = any(geom.get("type") == "plane" for geom in world.findall("geom"))
    if not has_ground:
        plant_ground = plant_world.find("geom[@name='ground']")
        if plant_ground is not None:
            world.append(plant_ground)

    asset = root.find("asset")
    plant_asset = plant_root.find("asset")
    if plant_asset is not None:
        if asset is None:
            asset = ET.Element("asset")
            root.insert(0, asset)
        for element in list(plant_asset):
            asset.append(element)
    world.append(plant_body)

    return ET.tostring(root, encoding="unicode")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aubo-dir", type=Path, required=True,
                        help="Manipulator-Mujoco AUBO i5 model directory")
    parser.add_argument("--ag95-dir", type=Path,
                        help="AG95 model directory (defaults to the linked AUBO repository's sibling model)")
    parser.add_argument("--output", type=Path, default=Path("tmp/strawberry_picker.xml"),
                        help="MJCF file to write")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parent / "dataset",
                        help="directory containing paired berry and leaf OBJ meshes")
    parser.add_argument("--primitive-berries", action="store_true",
                        help="use primitive berries instead of dataset OBJ meshes")
    parser.add_argument("--bundle-assets", action="store_true",
                        help="copy plant OBJ/MTL assets beside the XML")
    parser.add_argument("--seed", type=int, default=None, help="seed for the one generated plant")
    parser.add_argument("--plant-pos", type=lambda value: _vec3(value, "--plant-pos"),
                        default=(0.0, 0.0, 0.0), metavar="X Y Z")
    parser.add_argument("--robot-pos", type=lambda value: _vec3(value, "--robot-pos"),
                        default=(0.55, 0.0, 0.0), metavar="X Y Z")
    parser.add_argument("--validate", action="store_true", help="compile the saved scene with MuJoCo")
    parser.add_argument("--view", action="store_true", help="open the MuJoCo viewer")
    args = parser.parse_args()

    if not args.aubo_dir.is_dir():
        parser.error(f"--aubo-dir is not a directory: {args.aubo_dir}")
    ag95_dir = args.ag95_dir or (args.aubo_dir.parents[3]
                                  / "manipulator_mujoco" / "assets" / "robots" / "ag95")
    pairs = None if args.primitive_berries else find_berry_mesh_pairs(args.dataset_dir)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    copied_robot_dir = _copy_robot_package(args.aubo_dir, output, ag95_dir)
    _prepare_aubo_model(output.parent / copied_robot_dir, args.robot_pos)
    xml = build_picker_xml(
        args.aubo_dir,
        seed=args.seed,
        plant_position=args.plant_pos,
        robot_position=args.robot_pos,
        berry_mesh_pairs=pairs,
        copied_robot_dir=copied_robot_dir,
    )
    root = ET.fromstring(xml)
    if args.bundle_assets and pairs:
        _bundle_plant_meshes(root, output, args.dataset_dir)
        xml = ET.tostring(root, encoding="unicode")
    output.write_text(xml, encoding="utf-8")
    print(f"Wrote {output} with one generated plant and one AUBO i5")

    if args.validate or args.view:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(output.resolve()))
        print(f"Validated scene: {model.nbody} bodies, {model.ngeom} geoms")
    if args.view:
        import mujoco.viewer
        mujoco.viewer.launch(model, mujoco.MjData(model))


if __name__ == "__main__":
    main()

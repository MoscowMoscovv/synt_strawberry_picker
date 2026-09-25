"""Write a grid scene with optional portable copies of its dataset meshes."""

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def write_scene(xml: str, output: Path, mesh_root: Path | None = None) -> None:
    """Bundle referenced OBJ meshes relative to the XML when mesh_root is given.

    Each source is copied once even when several mesh assets use different scales.
    Meshes retain their dataset layout under <scene-name>_assets/ so equal filenames
    from different dataset subdirectories cannot collide.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if mesh_root is not None:
        root = ET.fromstring(xml)
        mesh_root = mesh_root.resolve()
        copied: set[Path] = set()
        for mesh in root.findall("./asset/mesh"):
            source = Path(mesh.attrib["file"]).resolve()
            relative = Path(f"{output.stem}_assets") / source.relative_to(mesh_root)
            destination = output.parent / relative
            if source not in copied:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source != destination.resolve():
                    shutil.copy2(source, destination)
                    # MuJoCo uses the OBJ geometry; retain exported MTLs for other tools.
                    material = source.with_suffix(".mtl")
                    if material.is_file():
                        shutil.copy2(material, destination.with_suffix(".mtl"))
                copied.add(source)
            mesh.set("file", relative.as_posix())
        xml = ET.tostring(root, encoding="unicode")
    output.write_text(xml, encoding="utf-8")

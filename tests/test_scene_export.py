import shutil
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco

from plant_generator.assets import find_berry_mesh_pairs
from plant_generator.mjcf_builder import build_grid_xml
from plant_generator.procedural import sample_plants
from plant_generator.scene_export import write_scene


TETRAHEDRON = """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
"""


class SceneExportTests(unittest.TestCase):
    def test_bundled_scene_loads_after_moving_without_original_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dataset = base / "dataset"
            # Two batches deliberately have the same mesh filenames.
            for batch in ("first", "second"):
                for folder, filename in (("berries", "strawberry_0001.obj"),
                                         ("leaves", "leaves_0001.obj")):
                    source = dataset / batch / folder / filename
                    source.parent.mkdir(parents=True, exist_ok=True)
                    source.write_text(TETRAHEDRON, encoding="utf-8")
                    source.with_suffix(".mtl").write_text("newmtl Material\n", encoding="utf-8")
            pairs = find_berry_mesh_pairs(dataset)
            xml = build_grid_xml(sample_plants(1, seed=7), berry_mesh_pairs=pairs)
            output = base / "export" / "grid.xml"
            write_scene(xml, output, mesh_root=dataset)

            root = ET.parse(output).getroot()
            references = [mesh.get("file") for mesh in root.findall("./asset/mesh")]
            self.assertGreater(len(references), 4)  # multiple mesh scales share each file
            self.assertTrue(all(not Path(ref).is_absolute() for ref in references))
            self.assertEqual(len(set(references)), 4)
            self.assertEqual(len(list(output.parent.rglob("*.obj"))), 4)
            self.assertEqual(len(list(output.parent.rglob("*.mtl"))), 4)

            shutil.move(str(output.parent), str(base / "relocated"))
            shutil.rmtree(dataset)
            model = mujoco.MjModel.from_xml_path(str(base / "relocated" / "grid.xml"))
            self.assertGreater(model.nmesh, 4)
            self.assertGreater(model.nbody, 2)

    def test_primitive_scene_bundle_needs_no_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "nested" / "grid.xml"
            xml = build_grid_xml([])
            write_scene(xml, output, mesh_root=base / "missing-dataset")
            self.assertFalse((output.parent / "grid_assets").exists())
            model = mujoco.MjModel.from_xml_path(str(output))
            self.assertEqual(model.nmesh, 0)

    def test_unbundled_scene_preserves_xml(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "grid.xml"
            xml = build_grid_xml([])
            write_scene(xml, output)
            self.assertEqual(output.read_text(encoding="utf-8"), xml)


if __name__ == "__main__":
    unittest.main()

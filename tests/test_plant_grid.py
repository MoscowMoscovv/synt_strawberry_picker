import unittest
import xml.etree.ElementTree as ET

import mujoco

from plant_generator.mjcf_builder import build_grid_xml
from plant_generator.procedural import sample_plants


class PlantGridTests(unittest.TestCase):
    def test_sampled_angles_are_bounded_and_balanced(self):
        for params in sample_plants(20, seed=42):
            fruit = [leaf for leaf in params.leaves if leaf.truss is not None]
            foliage = [leaf for leaf in params.leaves if leaf.truss is None]
            self.assertTrue(3 <= len(fruit) <= 5)
            self.assertTrue(3 <= len(foliage) <= 6)
            self.assertTrue(all(20 <= leaf.elevation_deg <= 45 for leaf in fruit))
            self.assertTrue(all(55 <= leaf.elevation_deg <= 80 for leaf in foliage))
            angles = sorted(leaf.azimuth_deg for leaf in params.leaves)
            gaps = [(angles[(i + 1) % len(angles)] - angle) % 360
                    for i, angle in enumerate(angles)]
            sector = 360 / len(angles)
            self.assertTrue(all(0.5 * sector <= gap <= 1.5 * sector for gap in gaps))


    def test_grid_preserves_original_and_compiles(self):
        plants = sample_plants(3, seed=7)
        xml = build_grid_xml(plants, spacing=0.7)
        self.assertEqual(xml, build_grid_xml(sample_plants(3, seed=7), spacing=0.7))
        root = ET.fromstring(xml)
        world = root.find("worldbody")
        crowns = list(world.findall("body"))
        self.assertEqual([body.get("name") for body in crowns], [
            "crown", "plant1_crown", "plant2_crown", "plant3_crown"
        ])
        self.assertEqual([body.get("pos") for body in crowns], [
            "0 0 0", "0.700000 0.000000 0", "0.000000 0.700000 0",
            "0.700000 0.700000 0"
        ])
        # MuJoCo has separate namespaces for bodies, joints, and geoms.
        for tag in ("body", "joint", "geom", "camera", "mesh"):
            names = [element.get("name") for element in root.iter(tag) if element.get("name")]
            self.assertEqual(len(names), len(set(names)), tag)
        model = mujoco.MjModel.from_xml_string(xml)
        self.assertGreater(model.nbody, 4)


if __name__ == "__main__":
    unittest.main()

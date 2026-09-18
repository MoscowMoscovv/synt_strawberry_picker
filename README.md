# synt_strawberry_picker

Procedural MuJoCo strawberry plant generator.

## Quick start

```bash
pip install -e ".[dev]"
python make_crown.py
```

This generates `tmp/crown_petioles.xml` and launches the MuJoCo viewer.

## Generate paired strawberry and leaf OBJ meshes

The procedural asset is `Strawberry/Strawberry/Procedual_strawberry.blend`. Use
Blender 4.3.x and run the batch driver with regular Python. It starts Blender
headlessly and evaluates the `Strawberry_procedual` Geometry Nodes object.
The Blender asset is a local dependency; its license does not permit
redistributing it with this repository.

```powershell
python batch_generate_strawberries.py `
  --blender 'C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe' `
  --count 10 --seed 42 `
  --output-dir generated_strawberries
```

This writes `N` fruit-only OBJs to `berries/`, `N` matching leaf-only OBJs to
`leaves/`, and `manifest.json` with their parameters and mesh counts. Matching
numbers form a pair in the asset's local coordinates, such as
`berries/strawberry_0001.obj` and `leaves/leaves_0001.obj`. The `--seed` makes
the variation repeatable. The generator varies the seed placement, noise W,
whole-object scale, and leaf scale. Use `--scale-range MIN MAX`, `--w-range MIN
MAX`, and `--leaf-scale-range MIN MAX` to adjust those ranges. Equal bounds fix
a value. The asset's saved `Main/Subdivide` value is 7, yielding roughly
300,000 fruit faces; use `--subdivide 5` for lighter meshes. The script refuses
to replace existing outputs unless `--overwrite` is given.

For individually specified models, use `--config strawberry_variants.example.json`
instead of `--count`. Each `parameters` key can be a unique input name, a
panel-qualified name such as `Divots/Seed`, or a socket ID such as `Input_2`.
Omitted inputs keep the values saved on the source object. Run with
`--list-parameters` to print all controls. Set `"part": "leaves"` for a
leaf-only model. The literal Blender input is `Leaves/On\off leaves` (one
backslash; doubled in JSON).

OBJ carries mesh geometry and a basic MTL material. The procedural Blender
shader is not baked into an OBJ texture; bake textures separately if needed.

## Inspect a strawberry in MuJoCo

Run the viewer with the project's environment (which includes MuJoCo):

```powershell
.\.venv\Scripts\python.exe view_strawberry_mujoco.py --leaf-count 1
```

The script creates one fruit mesh and the requested number of leaf meshes under
`dataset/seed_42_subdiv_5/` on the first run. Later runs reuse existing OBJs;
increasing `--leaf-count` generates only missing leaves. It writes `preview.xml`
and opens a static MuJoCo viewer. The fruit and leaves are visual meshes fixed
to the world, with no joints, collisions, or physics stepping. Close the viewer
window to end the script. Use `--check-only` to compile the scene without a GUI,
`--refresh` to regenerate cached meshes, or `--seed`, `--subdivide`, and
`--blender` to change the asset generation settings. Additional leaf meshes are
rotated around the fruit for inspection.

## Use dataset berries on the plant

`make_crown.py` currently has the literal `USE_DATASET_BERRIES = True`. Run the
plant as usual:

```powershell
.\.venv\Scripts\python.exe make_crown.py
```

With the literal set to `False`, the plant keeps its original ellipsoid berries.
With it set to `True`, the plant cycles through berry OBJ files under `dataset/`
and attaches each berry's matching generated leaf mesh at its top. The fruit
and leaves share the same scale and orientation along the berry's pedicel. An
invisible ellipsoid supplies collision and mass. The dataset must contain a
leaf OBJ for each berry: `berry.obj` pairs with `leaves_0001.obj`, while
`berries/strawberry_0001.obj` pairs with `leaves/leaves_0001.obj`. If the
dataset is empty, run `view_strawberry_mujoco.py --check-only` first to
generate a pair.

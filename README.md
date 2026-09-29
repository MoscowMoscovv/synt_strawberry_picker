# synt_strawberry_picker

Procedural MuJoCo strawberry plant generator.

## Generate scenes in Docker

The container includes Python 3.11, the dependencies in `requirements.txt`, and
[Blender 4.3.2](https://www.blender.org/download/releases/4-3/). Blender's official
Linux download is verified with its SHA-256 checksum during the build. This is a
CPU-based, headless generation workflow; use the MuJoCo viewer on your host.
On Windows, start Docker Desktop with Linux containers enabled.

From this repository directory, build the image and enter its Bash terminal:

```powershell
docker compose build
docker compose run --rm generator
```

Compose uses [bind mounts](https://docs.docker.com/engine/storage/bind-mounts/)
to connect these folders on your machine to the container:

| Local folder | Container folder | Purpose |
| --- | --- | --- |
| `dataset/` | `/app/dataset/` | Generated or existing paired OBJ meshes |
| `scenes/` | `/output/` | Exported MuJoCo scenes and bundled meshes |

The source asset at `Strawberry/Strawberry/Procedual_strawberry.blend`, its
textures, and its supplied license are copied into the image during the build.
No host asset mount is needed to generate meshes. Existing dataset meshes can
also be reused. Compose creates missing output and dataset mount directories.

Inside the container terminal (Bash), generate a dataset in a new subdirectory,
then export a plant scene:

```bash
python batch_generate_strawberries.py \
  --count 10 --seed 42 --subdivide 5 --max-faces 10000 \
  --output-dir /app/dataset/docker_seed42_mujoco

python generate_plant_grid.py \
  --count 8 --seed 42 --dataset-dir /app/dataset/docker_seed42_mujoco \
  --bundle-assets --validate --output /output/plant_grid.xml

exit
```

Skip the batch command when reusing a dataset and set `--dataset-dir` to its
directory. The batch generator refuses to replace files unless `--overwrite`
is given. `--count 8` creates eight varied plants plus the original plant.
For grids, use `--max-faces 10000` to limit each exported mesh to 10,000
triangles. Each distinct berry size needs a separate scaled mesh asset in
MuJoCo. Full-detail berries can therefore exhaust Docker's memory when many
plants are loaded, even when all OBJ files are present and valid. The face
limit simplifies visual geometry; plant collisions and mass still use the
existing primitive colliders. Lower `--subdivide` alone does not reduce the
asset's detailed seeds enough for large grids. Omit `--max-faces` for
full-detail mesh exports. Use a new dataset directory to keep both versions.

You will find `scenes/plant_grid.xml` and `scenes/plant_grid_assets/` on your
machine. `--bundle-assets` copies referenced OBJ meshes and their exported MTL
files beside the XML and writes relative mesh paths. Keep the XML and its asset
folder together when moving them. `--validate` compiles the saved scene in
MuJoCo without opening a window. Files in mounted folders persist after `exit`,
even though `--rm` removes the container. Other container files do not persist.
The source code is copied into the image; rebuild after changing it.

For a quick scene without Blender assets or a dataset, run this directly from
your host terminal:

```powershell
docker compose run --rm generator python generate_plant_grid.py --count 8 --seed 42 --primitive-berries --validate --output /output/primitive_grid.xml
```

On Linux, add `--user "$(id -u):$(id -g)" -e HOME=/tmp` after `run --rm` so
generated files belong to your user. Ensure both host mount folders exist
and are writable before starting. The image targets `linux/amd64` to match
Blender 4.3.2's Linux binary.

## Quick start

```bash
pip install -e ".[dev]"
python make_crown.py
```

This generates `tmp/crown_petioles.xml` and launches the MuJoCo viewer.

## Generate a grid of plants

```powershell
.\.venv\Scripts\python.exe generate_plant_grid.py --count 8 --seed 42 --view
```

This keeps the existing `PlantParams()` plant at the origin and adds eight
randomized plants in one scene on a 0.7 m grid. The scene is written to
`tmp/plant_grid.xml`; omit `--view` to generate it without opening MuJoCo.
Use `--spacing METRES` and `--output PATH` to change the layout and file, or
`--primitive-berries` to avoid loading dataset OBJ meshes. The same seed gives
the same plants and XML. Use `--dataset-dir PATH` to select a mesh dataset,
`--bundle-assets` for a portable scene, and `--validate` for a headless MuJoCo
compilation check. Plant counts, elevations, sizes, stiffness, truss
branches, and other sampling ranges are editable in `PlantRanges` in
`plant_generator/procedural.py`.

Stem azimuths occupy evenly spaced circular sectors with small random offsets.
Fruit stems start 20–45° above the ground, and leaf-only stems start 55–80°
above it. MuJoCo is Z-up, so the ground plane and grid are XY.

## Generate paired strawberry and leaf OBJ meshes

The procedural asset is `Strawberry/Strawberry/Procedual_strawberry.blend`. Use
Blender 4.3.x and run the batch driver with regular Python. It starts Blender
headlessly and evaluates the `Strawberry_procedual` Geometry Nodes object.
The Blender asset was created by [NorkAnimations](https://x.com/norkanimations)
and has a [separate license](Strawberry/Strawberry/License%20Agreement.txt).
It is included in this repository with the creator's redistribution permission.
The project's Apache 2.0 license does not apply to this third-party asset.
Its license restricts redistribution; including the files here does not grant
permission to redistribute them elsewhere.

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
300,000 fruit faces; use `--subdivide 5` for lighter meshes, or `--max-faces 10000`
to cap the exported triangle count for larger MuJoCo scenes. The script refuses
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

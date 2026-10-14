# Stem cutting status

## Implemented

- The composed picker scene adds two physical finger hitboxes using the shared geometry in `plant_generator/finger_hitboxes.py`.
- Runtime cutting requires the blue and orange hitboxes to contact the same named stem segment for three consecutive physics steps. It does not depend on whether the gripper is opening or closing.
- A cut selects the nearest eligible segment on the world `+Y` side of the contact. It disconnects that segment, removes its ball-joint spring and damping, and can attach the released subtree to the left finger at the cut-time pose.
- The hold weld is released when the controller detects that the gripper target is opening.
- `control_aubo_i5_gui.py --disable-stem-cutting` skips runtime cut and hold/release updates. It does not remove the cut-support joints and equalities from an already generated scene.
- Stem generation and the original stiffness-gradient formulas in `plant_generator/mjcf_builder.py` were left unchanged.

## Stability finding

The current cut-support representation is unstable in the composed scene. It adds three slide joints and an active `connect` equality to every stem segment, plus an inactive hold weld. The seed-42 picker scene develops a QACC blow-up even when runtime cutting is disabled. Removing those added joints, sites, and equalities from that same composed scene allowed 500 physics steps without the instability warning. The seeded plant also opens in the plant-only viewer.

The temporary comparison scene used during diagnosis is `tmp/strawberry_picker_seed42_without_cut_support.xml`. It retains the robot and seeded plant but omits the cut-support additions. The generator does not yet have a flag to create this variant.

## Next work

1. Replace or redesign the slide-joint plus stiff-connect approach before enabling cutting in normal scenes.
2. Preserve the existing stem builder, segment geometry, original stiffness gradients, hitbox positions and sizes, same-segment two-hitbox trigger, world `+Y` fragment selection, and gripper-open release behavior.
3. Add a scene-generation option for a no-cut-support baseline so stability can be checked without editing generated XML by hand.
4. Validate the replacement in a seeded plant-only model and the composed robot scene, checking QACC stability before exercising cut and release behavior.

## Useful commands

Generate the seeded picker scene:

```powershell
python create_picker_scene.py --aubo-dir Manipulator-Mujoco/manipulator_mujoco/assets/robots/aubo_i5 --primitive-berries --seed 42 --validate
```

Run the diagnostic scene without runtime cut handling:

```powershell
python control_aubo_i5_gui.py --scene tmp/strawberry_picker_seed42_without_cut_support.xml --debug --disable-stem-cutting
```

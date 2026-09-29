🍓 Procedural Strawberry Generator for Blender (Geometry Nodes)

This generator allows you to procedurally create and customize strawberries — shape, size, seeds, and leaves — all in one place.

It also supports generating a low-poly mesh for baking color and normal maps (after proper UV unwrapping).

Two example baked strawberry models are already included in the file for reference.

🔧 HOW TO USE:

Create a base mesh (for example, a cube), and apply the Geometry Nodes modifier using the group 'Procedural_Strawberry'.

Do not use overly high-poly meshes — this node tree handles detailing internally.

— FEATURES & CONTROLS —

🟪 MAIN CONTROLS

- Seeds Collection — a hidden collection containing seed instances (do not change it)

- Random rotation — enables random rotation for seeds (usually not necessary)

- Subdivide — mesh subdivision level

- Smooth — mesh smoothing

- Material — procedural material slot (leave unchanged)

- Scale all — scales the entire strawberry

- Scale (no seeds) — scales only the mesh (useful for adjusting seed gaps)

🟧 NOISE SHAPE PANEL — tweak strawberry form

- W — random shape variation

- Noise scale — scale of the deformation

- Noise strength — strength of the deformation

🟩 DIVOTS PANEL — controls seed placement

- Seed level — vertical spread level of seeds (doesn’t auto-scale with strawberry size!)

- Seed — profile for seed scattering

- Divot scale — seed size

- Seed min / Seed max — min and max size of individual seeds

- Distance min — space between seeds

- Density factor — seed amount multiplier (can be masked)

🟨 HOLES PANEL — controls for seed dents

- Hole strength — depth of seed dents

- Hole scale — size of dents

🟫 LEAVES PANEL — leaf settings

- On/Off leaves — enable or disable leaves

- Subdivision leaves — level of detail for leaves

- Scale leaves — leaf size

- Position leaves — vertical offset

🟥 MATERIAL PANEL

- Down yellow color fac — adds a yellow gradient at the bottom of the strawberry

- Red part blur — softens transition between red and yellow

🎨 MANUAL MATERIAL TWEAKS:

You can fine-tune the colors manually in the material editor by adjusting:

- 'Strawberry color'

- 'Seeds color'

- 'Seeds gradient color'

- 'Reflections'

- 'Normal detail' (bump strength)

🟦 BAKE PANEL — for baking preparation

- LOW mesh — enables low-poly version of the mesh (apply modifiers before UV unwrap)

- Subdivide — adds more geometry to low-poly mesh

- Cage — enables a projection cage for baking (optional in Blender)

📁 FILE INCLUDES:

• Geometry Node group 'Procedural_Strawberry'

• Two example baked strawberries

• Ready-to-edit .blend file


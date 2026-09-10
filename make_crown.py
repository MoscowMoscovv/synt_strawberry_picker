"""Crown + segmented petioles + segmented truss stems with stiffness gradient.
  Berries on separate bodies with ball joints.

Run:  python make_crown.py
"""

import math
import numpy as np
import mujoco
import mujoco.viewer


# ── helpers ──────────────────────────────────────────────────────

def exp_stiffness(k_base, decay, i, n):
    """Exponential decay stiffness: k_base * exp(-decay * i / (n-1))."""
    return k_base * math.exp(-decay * i / max(n - 1, 1))


def logistic_stiffness(k_base, alpha, i_break, k_min, i, n):
    """Logistic stiffness gradient with configurable floor.

    k(i) = k_min + (k_base - k_min) / (1 + exp(alpha * (i - i_break) / (n - 1)))

    At i=0: k ≈ k_base (stiff root)
    At i=n-1: k ≈ k_min (soft tip, set by parameter)
    """
    t = (i - i_break) / max(n - 1, 1)
    return k_min + (k_base - k_min) / (1.0 + math.exp(alpha * t))


def taper_radius(r_base, taper_ratio, i, n):
    t = i / max(n - 1, 1)
    return r_base * (1.0 - t * (1.0 - taper_ratio))


def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def quat_from_rotmat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2 * math.sqrt(tr + 1);
        w, x, y, z = .25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]);
        w = (R[2, 1] - R[1, 2]) / s;
        x = .25 * s;
        y = (R[0, 1] + R[1, 0]) / s;
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2 * math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]);
        w = (R[0, 2] - R[2, 0]) / s;
        x = (R[0, 1] + R[1, 0]) / s;
        y = .25 * s;
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2 * math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]);
        w = (R[1, 0] - R[0, 1]) / s;
        x = (R[0, 2] + R[2, 0]) / s;
        y = (R[1, 2] + R[2, 1]) / s;
        z = .25 * s
    q = np.array([w, x, y, z]);
    return q / np.linalg.norm(q)


def quat_leaf(az, elev):
    return quat_from_rotmat(_Rz(az) @ _Ry(-elev))


def q2s(q):
    return f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}"


def ellipsoid_mass(a, b, c, density):
    return (4 / 3) * math.pi * a * b * c * density


def box_mass(lx, ly, lz, density):
    return lx * ly * lz * density


# ── params ───────────────────────────────────────────────────────

N_PET_SEG = 12  # petiole segments — more nodes for smoother compliance taper
N_STEM_SEG = 3  # truss stem segments
PET_K_BASE = 1.0      # exponential k_base — stiffness at root (berry petioles)
PET_K_DECAY = 2.0     # exponential decay rate (berry petioles)
PET_KF_BASE = 2.5     # logistic k_base — stiffness at root (leaf petioles)
PET_KF_ALPHA = 9.0   # logistic sharpness (leaf petioles)
PET_KF_IBREAK = 6.5   # logistic inflection point, segment index (leaf petioles)
PET_KF_MIN = 0.03      # minimum stiffness at tip (leaf petioles)
DAMPING_RATIO = 0.4
PET_R_BASE = 0.0015;
PET_TAPER = 0.6;
PET_LEN = 0.18

STEM_K_BASE = 0.15    # exponential k_base — stiffness at stem root
STEM_K_DECAY = 2.5    # exponential decay rate
STEM_R_BASE = 0.001;
STEM_TAPER = 0.5

DENSITY = 450.0;
LEAF_DENSITY = 900.0  # lighter than stem — leaf blade density
STEM_RGBA = "0.38 0.40 0.20 1";
LEAF_RGBA = "0.25 0.55 0.22 1";
BERRY_RGBA = "0.85 0.1 0.08 1"
LEAF_BASE_K = 0.2     # leaf-base joint stiffness (N·m/rad)
LEAF_BASE_D = 0.001    # leaf-base joint damping
BERRY_K = 0.008;
BERRY_D = 0.004
BERRY_DENSITY = 1450.0

CT = {"ground": 1, "structure": 2, "leaf": 4, "berry": 8}
CA = {"ground": 15, "structure": 25, "leaf": 25, "berry": 15}

LEAVES = [
    (0.0, math.radians(55)),  # 0: truss A
    (math.radians(137), math.radians(45)),  # 1: truss B
    (math.radians(274), math.radians(35)),  # 2: truss C
    (math.radians(90), math.radians(60)),  # 3: truss D
    (math.radians(200), math.radians(70)),  # 4: leaf only
    (math.radians(310), math.radians(80)),  # 5: leaf only
    (math.radians(45), math.radians(75)),  # 6: leaf only
]

# Per-truss config: (stem_len, plane_az, branches)
# branch = (frac, ped_len, droop_rad, berry_r_scale, has_sub_branch, sub_branch_params_or_None)
# sub_branch_params = (sub_len, sub_droop, sub_berry_r_scale)
TRUSS_CFG = {
    0: {"stem_len": 0.07, "plane_az": 0.0, "branches": [
        (0.45, 0.035, math.radians(40), 1.0, False, None),
        (0.70, 0.030, math.radians(40), 1.0, False, None),
    ], "tip_berry_r_scale": 1.0, "tip_ped_len": 0.025},
    1: {"stem_len": 0.06, "plane_az": 0.0, "branches": [
        (0.40, 0.032, math.radians(25), 1.0, False, None),
        (0.70, 0.028, math.radians(20), 0.9, True,
         (0.022, math.radians(35), 0.81)),
        (0.85, 0.025, math.radians(30), 0.85, False, None),
    ], "tip_berry_r_scale": 1.0, "tip_ped_len": 0.020},
    2: {"stem_len": 0.055, "plane_az": 0.0, "branches": [
        (0.45, 0.025, math.radians(18), 1.0, True,
         (0.018, math.radians(30), 0.85)),
        (0.75, 0.022, math.radians(28), 1.0, False, None),
    ], "tip_berry_r_scale": 1.0, "tip_ped_len": 0.018},
    3: {"stem_len": 0.065, "plane_az": 0.0, "branches": [
        (0.40, 0.030, math.radians(35), 1.0, False, None),
        (0.65, 0.025, math.radians(35), 1.0, False, None),
    ], "tip_berry_r_scale": 1.0, "tip_ped_len": 0.020},
}

BERRY_R_BASE = 0.012

# ── trifoliate leaf geometry ──
LEAFLET_L = 0.095   # leaflet length (m)
LEAFLET_W = 0.040   # leaflet width (m)
LEAFLET_T = 0.003   # leaflet thickness (m)
LEAF_SPREAD = math.radians(40)  # side leaflet splay angle

# ── build ─────────────────────────────────────────────────────────

L: list[str] = []
L.append('<mujoco model="crown_petioles_truss">')
L.append('  <compiler angle="degree"/>')
L.append('  <option timestep="0.002" integrator="implicitfast"/>')
L.append('  <default><joint limited="false"/></default>')
L.append('  <worldbody>')
L.append(
    f'    <geom name="ground" type="plane" size="10 10 0.1" contype="{CT["ground"]}" conaffinity="{CA["ground"]}" rgba="0.3 0.25 0.2 1" friction="1.0 0.005 0.0001"/>')
L.append('    <camera name="robot_view" pos="0 0.20 0.35" xyaxes="1 0 0 0 0.423 -0.906" fovy="45"/>')
L.append('')

L.append(f'    <body name="crown" pos="0 0 0">')
L.append(
    f'      <geom name="crown0" type="cylinder" pos="0 0 0" size="0.02 0.01" contype="{CT["structure"]}" conaffinity="{CA["structure"]}" rgba="{STEM_RGBA}" density="{DENSITY:.1f}"/>')

pet_seg_len = PET_LEN / N_PET_SEG

# ── petiole chains ──
for li, (az, elev) in enumerate(LEAVES):
    q = quat_leaf(az, elev)
    radii = [taper_radius(PET_R_BASE, PET_TAPER, i, N_PET_SEG) for i in range(N_PET_SEG)]
    if li in TRUSS_CFG:
        stiffs = [exp_stiffness(PET_K_BASE, PET_K_DECAY, i, N_PET_SEG)/2 for i in range(N_PET_SEG)]
    else:
        stiffs = [logistic_stiffness(PET_KF_BASE, PET_KF_ALPHA, PET_KF_IBREAK, PET_KF_MIN, i, N_PET_SEG)/2 for i in range(N_PET_SEG)]
    damps = [DAMPING_RATIO * k for k in stiffs]

    L.append(f'      <body name="leaf{li}_seg0" pos="0 0 {PET_R_BASE:.6f}" quat="{q2s(q)}">')
    L.append(
        f'        <joint name="leaf{li}_seg0_joint" type="ball" stiffness="{stiffs[0]:.6f}" damping="{damps[0]:.6f}"/>')
    L.append(
        f'        <geom name="leaf{li}_seg0_cap" type="capsule" fromto="0 0 0 {pet_seg_len:.6f} 0 0" size="{radii[0]:.6f}" contype="{CT["leaf"]}" conaffinity="{CA["leaf"]}" rgba="{STEM_RGBA}" density="{DENSITY:.1f}"/>')

    for si in range(1, N_PET_SEG - 1):
        L.append(f'        <body name="leaf{li}_seg{si}" pos="{pet_seg_len:.6f} 0 0">')
        L.append(
            f'          <joint name="leaf{li}_seg{si}_joint" type="ball" stiffness="{stiffs[si]:.6f}" damping="{damps[si]:.6f}"/>')
        L.append(
            f'          <geom name="leaf{li}_seg{si}_cap" type="capsule" fromto="0 0 0 {pet_seg_len:.6f} 0 0" size="{radii[si]:.6f}" contype="{CT["leaf"]}" conaffinity="{CA["leaf"]}" rgba="{STEM_RGBA}" density="{DENSITY:.1f}"/>')

    si = N_PET_SEG - 1
    L.append(f'        <body name="leaf{li}_seg{si}" pos="{pet_seg_len:.6f} 0 0">')
    L.append(
        f'          <joint name="leaf{li}_seg{si}_joint" type="ball" stiffness="{stiffs[si]:.6f}" damping="{damps[si]:.6f}"/>')
    L.append(
        f'          <geom name="leaf{li}_seg{si}_cap" type="capsule" fromto="0 0 0 {pet_seg_len:.6f} 0 0" size="{radii[si]:.6f}" contype="{CT["leaf"]}" conaffinity="{CA["leaf"]}" rgba="{STEM_RGBA}" density="{DENSITY:.1f}"/>')

    # trifoliate leaf — only for non-truss petioles
    # leaf base body: elastic ball joint at petiole tip, leaflets rigidly attached
    if li not in TRUSS_CFG:
        L.append(f'        <body name="leaf{li}_base" pos="{pet_seg_len:.6f} 0 0">')
        L.append(f'          <joint name="leaf{li}_base_joint" type="ball" stiffness="{LEAF_BASE_K:.6f}" damping="{LEAF_BASE_D:.6f}"/>')
        # center leaflet (straight ahead along petiole)
        L.append(
            f'          <geom name="leaf{li}_blade_c" type="box" pos="{LEAFLET_L / 2:.6f} 0 0" size="{LEAFLET_L / 2:.6f} {LEAFLET_W / 2:.6f} {LEAFLET_T / 2:.6f}" contype="{CT["leaf"]}" conaffinity="{CA["leaf"]}" rgba="{LEAF_RGBA}" density="{LEAF_DENSITY:.1f}"/>')
        # left leaflet (−spread)
        q_l = q2s(quat_from_rotmat(_Rz(-LEAF_SPREAD)))
        lx = LEAFLET_L / 2 * math.cos(LEAF_SPREAD)
        ly = -LEAFLET_L / 2 * math.sin(LEAF_SPREAD)
        L.append(
            f'          <geom name="leaf{li}_blade_l" type="box" pos="{lx:.6f} {ly:.6f} 0" quat="{q_l}" size="{LEAFLET_L / 2:.6f} {LEAFLET_W / 2:.6f} {LEAFLET_T / 2:.6f}" contype="{CT["leaf"]}" conaffinity="{CA["leaf"]}" rgba="{LEAF_RGBA}" density="{LEAF_DENSITY:.1f}"/>')
        # right leaflet (+spread)
        q_r = q2s(quat_from_rotmat(_Rz(LEAF_SPREAD)))
        rx = LEAFLET_L / 2 * math.cos(LEAF_SPREAD)
        ry = LEAFLET_L / 2 * math.sin(LEAF_SPREAD)
        L.append(
            f'          <geom name="leaf{li}_blade_r" type="box" pos="{rx:.6f} {ry:.6f} 0" quat="{q_r}" size="{LEAFLET_L / 2:.6f} {LEAFLET_W / 2:.6f} {LEAFLET_T / 2:.6f}" contype="{CT["leaf"]}" conaffinity="{CA["leaf"]}" rgba="{LEAF_RGBA}" density="{LEAF_DENSITY:.1f}"/>')
        L.append(f'        </body>')

    # ── truss stems (segmented) ──
    if li in TRUSS_CFG:
        cfg = TRUSS_CFG[li]
        stem_len = cfg["stem_len"]
        stem_seg_len = stem_len / N_STEM_SEG
        stem_radii = [taper_radius(STEM_R_BASE, STEM_TAPER, i, N_STEM_SEG) for i in range(N_STEM_SEG)]
        stem_stiffs = [exp_stiffness(STEM_K_BASE, STEM_K_DECAY, i, N_STEM_SEG) for i in range(N_STEM_SEG)]
        stem_damps = [DAMPING_RATIO * k for k in stem_stiffs]
        BERRY_R = BERRY_R_BASE

        # First stem segment (child of last petiole segment)
        L.append(f'          <body name="leaf{li}_stem_seg0" pos="{pet_seg_len:.6f} 0 0">')
        L.append(f'          <joint name="leaf{li}_stem_seg0_joint" type="ball" stiffness="{stiffs[-1]:.6f}" damping="0.0"/>')
        L.append(f'          <geom name="leaf{li}_stem_seg0" type="capsule" fromto="0 0 0 {stem_seg_len:.6f} 0 0 " size="{stem_radii[0]:.6f}" contype="{CT["structure"]}" conaffinity="{CA["structure"]}" rgba="{STEM_RGBA}" density="{DENSITY:.1f}"/>')

        # Middle stem segments
        for si in range(1, N_STEM_SEG - 1):
            L.append(f'            <body name="leaf{li}_stem_seg{si}" pos="{stem_seg_len:.6f} 0 0">')
            L.append(
                f'              <joint name="leaf{li}_stem_seg{si}_joint" type="ball" stiffness="{stem_stiffs[si]:.6f}" damping="{stem_damps[si]:.6f}"/>')
            L.append(
                f'              <geom name="leaf{li}_stem_seg{si}" type="capsule" fromto="0 0 0 {stem_seg_len:.6f} 0 0 " size="{stem_radii[si]:.6f}" contype="{CT["structure"]}" conaffinity="{CA["structure"]}" rgba="{STEM_RGBA}" density="{DENSITY:.1f}"/>')

        # Last stem segment — branches attach here
        si = N_STEM_SEG - 1
        L.append(f'            <body name="leaf{li}_stem_seg{si}" pos="{stem_seg_len:.6f} 0 0">')
        L.append(
            f'              <joint name="leaf{li}_stem_seg{si}_joint" type="ball" stiffness="{stem_stiffs[si]:.6f}" damping="{stem_damps[si]:.6f}"/>')
        L.append(
            f'              <geom name="leaf{li}_stem_seg{si}" type="capsule" fromto="0 0 0 {stem_seg_len:.6f} 0 0 " size="{stem_radii[si]:.6f}" contype="{CT["structure"]}" conaffinity="{CA["structure"]}" rgba="{STEM_RGBA}" density="{DENSITY:.1f}"/>')

        PED_R = 0.0007;
        PED_K = 0.008;
        PED_D = 0.003

        # Branches — all pedicels lie in a single vertical plane at plane_az
        plane_az = cfg["plane_az"]
        bi = 0
        for (frac, ped_len, droop, br_scale, has_sub, sub_params) in cfg["branches"]:
            # Which stem segment does this branch attach to?
            # frac is fraction along the FULL stem
            attach_seg = min(int(frac * N_STEM_SEG), N_STEM_SEG - 1)
            local_z = frac * stem_len - attach_seg * stem_seg_len
            # For simplicity, attach all branches to last segment with local offset
            # (since all fracs > 0.4 and N_STEM_SEG=3, last seg covers 0.66-1.0)
            local_z_in_seg = frac * stem_len - (N_STEM_SEG - 1) * stem_seg_len

            # Pedicel direction: perpendicular to stem (x-axis), in the y-z plane
            dx = 0
            dy = ped_len * math.cos(droop) * math.cos(plane_az)
            dz = -math.sin(droop) * ped_len

            L.append(f'              <body name="leaf{li}_branch{bi}" pos="{local_z_in_seg:.6f} 0 0">')
            L.append(
                f'                <joint name="leaf{li}_branch{bi}_joint" type="ball" stiffness="{PED_K:.6f}" damping="{PED_D:.6f}"/>')
            L.append(
                f'                <geom name="leaf{li}_branch{bi}_stem" type="capsule" fromto="0 0 0 {dx:.6f} {dy:.6f} {dz:.6f}" size="{PED_R:.6f}" contype="{CT["structure"]}" conaffinity="{CA["structure"]}" rgba="{STEM_RGBA}" density="{DENSITY:.1f}"/>')

            if has_sub and sub_params is not None:
                sub_len, sub_droop, sub_br_scale = sub_params
                sdx = 0
                sdy = sub_len * math.cos(sub_droop) * math.cos(plane_az)
                sdz = -math.sin(sub_droop) * sub_len
                L.append(f'                <body name="leaf{li}_branch{bi}b" pos="{dx:.6f} {dy:.6f} {dz:.6f}">')
                L.append(
                    f'                  <joint name="leaf{li}_branch{bi}b_joint" type="ball" stiffness="{PED_K * 0.7:.6f}" damping="{PED_D * 0.7:.6f}"/>')
                L.append(
                    f'                  <geom name="leaf{li}_branch{bi}b_stem" type="capsule" fromto="0 0 0 {sdx:.6f} {sdy:.6f} {sdz:.6f}" size="{PED_R:.6f}" contype="{CT["structure"]}" conaffinity="{CA["structure"]}" rgba="{STEM_RGBA}" density="{DENSITY:.1f}"/>')
                # berry on its own body
                L.append(f'                  <body name="leaf{li}_berry{bi}b" pos="{sdx:.6f} {sdy:.6f} {sdz:.6f}">')
                L.append(
                    f'                    <joint name="leaf{li}_berry{bi}b_joint" type="ball" stiffness="{BERRY_K:.6f}" damping="{BERRY_D:.6f}"/>')
                L.append(
                    f'                    <geom name="leaf{li}_berry{bi}b_geom" type="ellipsoid" size="{BERRY_R * sub_br_scale * 1.15:.6f} {BERRY_R * sub_br_scale:.6f} {BERRY_R * sub_br_scale:.6f}" contype="{CT["berry"]}" conaffinity="{CA["berry"]}" rgba="{BERRY_RGBA}" density="{BERRY_DENSITY}"/>')
                L.append(f'                  </body>')
                L.append(f'                </body>')
            else:
                # berry on its own body at end of pedicel
                L.append(f'                <body name="leaf{li}_berry{bi}" pos="{dx:.6f} {dy:.6f} {dz:.6f}">')
                L.append(
                    f'                  <joint name="leaf{li}_berry{bi}_joint" type="ball" stiffness="{BERRY_K:.6f}" damping="{BERRY_D:.6f}"/>')
                L.append(
                    f'                  <geom name="leaf{li}_berry{bi}_geom" type="ellipsoid" size="{BERRY_R * br_scale * 1.15:.6f} {BERRY_R * br_scale:.6f} {BERRY_R * br_scale:.6f}" contype="{CT["berry"]}" conaffinity="{CA["berry"]}" rgba="{BERRY_RGBA}" density="{BERRY_DENSITY}"/>')
                L.append(f'                </body>')

            L.append(f'              </body>')
            bi += 1

        # Tip berry at end of last stem segment
        tip_ped_len = cfg["tip_ped_len"]
        tip_r_scale = cfg["tip_berry_r_scale"]
        L.append(f'              <body name="leaf{li}_ped_tip" pos="{stem_seg_len:.6f} 0 0">')
        L.append(
            f'                <joint name="leaf{li}_ped_tip_joint" type="ball" stiffness="{PED_K:.6f}" damping="{PED_D:.6f}"/>')
        L.append(
            f'                <geom name="leaf{li}_ped_tip_cap" type="capsule" fromto="0 0 0 {tip_ped_len:.6f} 0 0" size="{PED_R:.6f}" contype="{CT["structure"]}" conaffinity="{CA["structure"]}" rgba="{STEM_RGBA}" density="{DENSITY:.1f}"/>')
        L.append(f'                <body name="leaf{li}_berry_tip" pos="{tip_ped_len:.6f} 0 0">')
        L.append(
            f'                  <joint name="leaf{li}_berry_tip_joint" type="ball" stiffness="{BERRY_K:.6f}" damping="{BERRY_D:.6f}"/>')
        L.append(
            f'                  <geom name="leaf{li}_berry_tip_geom" type="ellipsoid" size="{BERRY_R * tip_r_scale * 1.15:.6f} {BERRY_R * tip_r_scale:.6f} {BERRY_R * tip_r_scale:.6f}" contype="{CT["berry"]}" conaffinity="{CA["berry"]}" rgba="{BERRY_RGBA}" density="{BERRY_DENSITY}"/>')
        L.append(f'                </body>')
        L.append(f'              </body>')

        # Close last stem segment
        L.append(f'            </body>')  # stem_seg last
        for si in range(N_STEM_SEG - 2, 0, -1):
            L.append(f'            </body>')
        L.append(f'          </body>')  # stem_seg0

    L.append('        </body>')  # last petiole seg
    for si in range(N_PET_SEG - 2, 0, -1):
        L.append('        </body>')
    L.append('      </body>')  # petiole seg0

L.append('    </body>')  # crown
L.append('  </worldbody>')
L.append('</mujoco>')

xml = '\n'.join(L)
xml_path = "tmp/crown_petioles.xml"
with open(xml_path, 'w') as f:
    f.write(xml)
print(f"Wrote {xml_path}  ({len(xml)} bytes)")

# ── weight report ──
leaflet_mass = box_mass(LEAFLET_L, LEAFLET_W, LEAFLET_T, LEAF_DENSITY)
print(f"Leaflet ({LEAFLET_L*100:.0f}x{LEAFLET_W*100:.0f}x{LEAFLET_T*1000:.0f} cm, density={LEAF_DENSITY:.0f}): {leaflet_mass*1000:.2f} g each, {3*leaflet_mass*1000:.2f} g per trifoliate")

for name, br in [("R=12mm", 0.012), ("R=11mm", 0.011), ("R=10mm", 0.010)]:
    m = ellipsoid_mass(br * 1.15, br, br, 950)
    print(f"Berry {name} (ellipsoid, density=950): {m * 1000:.2f} g")

# stiffness report
print(f"\nBerry petiole stiffness gradient (K_BASE={PET_K_BASE}, decay={PET_K_DECAY}):")
for i in range(N_PET_SEG):
    k = exp_stiffness(PET_K_BASE, PET_K_DECAY, i, N_PET_SEG)
    print(f"  seg{i}: k={k:.4f} N·m/rad, r={taper_radius(PET_R_BASE, PET_TAPER, i, N_PET_SEG) * 1000:.2f} mm")

print(f"\nLeaf petiole stiffness gradient (K_BASE={PET_KF_BASE}, alpha={PET_KF_ALPHA}, i_break={PET_KF_IBREAK}, k_min={PET_KF_MIN}):")
for i in range(N_PET_SEG):
    k = logistic_stiffness(PET_KF_BASE, PET_KF_ALPHA, PET_KF_IBREAK, PET_KF_MIN, i, N_PET_SEG)
    print(f"  seg{i}: k={k:.4f} N·m/rad, r={taper_radius(PET_R_BASE, PET_TAPER, i, N_PET_SEG) * 1000:.2f} mm")

print(f"\nTruss stem stiffness gradient (K_BASE={STEM_K_BASE}, decay={STEM_K_DECAY}):")
for i in range(N_STEM_SEG):
    k = exp_stiffness(STEM_K_BASE, STEM_K_DECAY, i, N_STEM_SEG)
    print(f"  seg{i}: k={k:.4f} N·m/rad, r={taper_radius(STEM_R_BASE, STEM_TAPER, i, N_STEM_SEG) * 1000:.2f} mm")

print("\nLaunching viewer…")
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
for _ in range(500):
    mujoco.mj_step(model, data)
mujoco.viewer.launch(model, data)
print("Viewer closed.")

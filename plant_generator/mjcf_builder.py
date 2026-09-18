"""Build MuJoCo XML from PlantParams."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .geometry import (
    exp_stiffness,
    quat_from_rotmat,
    quat_leaf,
    q2s,
    taper_radius,
    _Rz,
)
from .params import PlantParams


def _obj_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the position bounds needed to fit an OBJ to a berry body."""
    low = np.full(3, math.inf)
    high = np.full(3, -math.inf)
    for line in path.open(encoding="utf-8", errors="replace"):
        if line.startswith("v "):
            xyz = np.fromiter((float(value) for value in line.split()[1:4]), dtype=float, count=3)
            if len(xyz) != 3 or not np.all(np.isfinite(xyz)):
                raise ValueError(f"Invalid vertex in {path}")
            low = np.minimum(low, xyz)
            high = np.maximum(high, xyz)
    if not np.all(np.isfinite(low)) or np.any(high <= low):
        raise ValueError(f"Empty or flat berry mesh: {path}")
    return low, high


def _berry_alignment(stem_direction: tuple[float, float, float]) -> tuple[np.ndarray, str]:
    """Aim a berry's local -Z (from its top toward its tip) along its pedicel."""
    direction = np.asarray(stem_direction, dtype=float)
    direction /= np.linalg.norm(direction)
    z_axis = -direction
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(reference, z_axis)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])
    x_axis = reference - np.dot(reference, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return rotation, q2s(quat_from_rotmat(rotation))


def build_xml(params: PlantParams, berry_mesh_pairs: list[tuple[Path, Path]] | None = None) -> str:
    L: list[str] = []
    ct = params.collision
    cr = params.crown
    berry_sources = [
        (berry_path.resolve(), leaf_path.resolve(), *_obj_bounds(berry_path))
        for berry_path, leaf_path in berry_mesh_pairs or []
    ]
    asset_lines: list[str] = []
    asset_names: dict[tuple[Path, float], str] = {}
    berry_index = 0

    def berry_geoms(name: str, radius_scale: float, indent: str,
                    stem_direction: tuple[float, float, float]) -> list[str]:
        nonlocal berry_index
        radius = berry.r_base * radius_scale
        if not berry_sources:
            return [
                f'{indent}<geom name="{name}" type="ellipsoid" '
                f'size="{radius * berry.oblateness:.6f} {radius:.6f} {radius:.6f}" '
                f'contype="{ct.berry[0]}" conaffinity="{ct.berry[1]}" '
                f'rgba="{params.berry_rgba}" density="{berry.density}"/>'
            ]

        path, leaf_path, low, high = berry_sources[berry_index % len(berry_sources)]
        berry_index += 1
        span = high - low
        mesh_scale = 2 * radius / max(span[0], span[1])
        key = (path, mesh_scale)
        asset_name = asset_names.get(key)
        if asset_name is None:
            asset_name = f"dataset_berry_{len(asset_names)}"
            asset_names[key] = asset_name
            asset_lines.append(
                f'    <mesh name="{asset_name}" file="{path.as_posix()}" '
                f'scale="{mesh_scale:.9g} {mesh_scale:.9g} {mesh_scale:.9g}"/>'
            )
        leaf_key = (leaf_path, mesh_scale)
        leaf_asset_name = asset_names.get(leaf_key)
        if leaf_asset_name is None:
            leaf_asset_name = f"dataset_berry_leaves_{len(asset_names)}"
            asset_names[leaf_key] = leaf_asset_name
            asset_lines.append(
                f'    <mesh name="{leaf_asset_name}" file="{leaf_path.as_posix()}" '
                f'scale="{mesh_scale:.9g} {mesh_scale:.9g} {mesh_scale:.9g}"/>'
            )

        top_center = np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, high[2]])
        rotation, berry_quat = _berry_alignment(stem_direction)
        visual_pos = -(rotation @ (top_center * mesh_scale))
        collider_pos = rotation @ np.array([0.0, 0.0, -span[2] * mesh_scale / 2])
        half_size = span * mesh_scale / 2

        def vec(values: np.ndarray) -> str:
            return " ".join(f"{value:.9g}" for value in values)

        return [
            f'{indent}<geom name="{name}" type="mesh" mesh="{asset_name}" '
            f'pos="{vec(visual_pos)}" quat="{berry_quat}" '
            f'contype="0" conaffinity="0" mass="0" rgba="{params.berry_rgba}"/>',
            f'{indent}<geom name="{name}_leaves" type="mesh" mesh="{leaf_asset_name}" '
            f'pos="{vec(visual_pos)}" quat="{berry_quat}" '
            f'contype="0" conaffinity="0" mass="0" rgba="{params.leaf_rgba}"/>',
            f'{indent}<geom name="{name}_collision" type="ellipsoid" '
            f'pos="{vec(collider_pos)}" quat="{berry_quat}" '
            f'size="{vec(half_size)}" '
            f'contype="{ct.berry[0]}" conaffinity="{ct.berry[1]}" '
            f'rgba="0 0 0 0" density="{berry.density}"/>',
        ]

    L.append(f'<mujoco model="{params.model_name}">')
    L.append('  <compiler angle="degree"/>')
    L.append(f'  <option timestep="{params.timestep}" integrator="{params.integrator}"/>')
    L.append('  <default><joint limited="false"/></default>')
    asset_insert_index = len(L)
    L.append('  <worldbody>')
    L.append(
        f'    <geom name="ground" type="plane" size="10 10 0.1" '
        f'contype="{ct.ground[0]}" conaffinity="{ct.ground[1]}" '
        f'rgba="0.3 0.25 0.2 1" friction="1.0 0.005 0.0001"/>')
    L.append('    <camera name="robot_view" pos="0 0.20 0.35" xyaxes="1 0 0 0 0.423 -0.906" fovy="45"/>')
    L.append('')

    # Crown body
    L.append(f'    <body name="crown" pos="0 0 0">')
    L.append(
        f'      <geom name="crown0" type="cylinder" pos="0 0 0" '
        f'size="{cr.radius:.6f} {cr.half_height:.6f}" '
        f'contype="{ct.structure[0]}" conaffinity="{ct.structure[1]}" '
        f'rgba="{params.stem_rgba}" density="{cr.density:.1f}"/>')

    # ── petiole chains ──
    for li, leaf in enumerate(params.leaves):
        profile = params.berry_profile if leaf.truss else params.leaf_profile
        az_rad = math.radians(leaf.azimuth_deg)
        elev_rad = math.radians(leaf.elevation_deg)
        q = quat_leaf(az_rad, elev_rad)
        n_seg = profile.n_seg
        seg_len = profile.length / n_seg
        radii = [taper_radius(profile.r_base, profile.taper, i, n_seg) for i in range(n_seg)]
        stiffs = [profile.stiffness(i, n_seg) for i in range(n_seg)]
        damps = [profile.damping_ratio * k for k in stiffs]

        L.append(f'      <body name="leaf{li}_seg0" pos="0 0 {profile.r_base:.6f}" quat="{q2s(q)}">')
        L.append(
            f'        <joint name="leaf{li}_seg0_joint" type="ball" '
            f'stiffness="{stiffs[0]:.6f}" damping="{damps[0]:.6f}"/>')
        L.append(
            f'        <geom name="leaf{li}_seg0_cap" type="capsule" fromto="0 0 0 {seg_len:.6f} 0 0" '
            f'size="{radii[0]:.6f}" contype="{ct.leaf[0]}" conaffinity="{ct.leaf[1]}" '
            f'rgba="{params.stem_rgba}" density="{cr.density:.1f}"/>')

        for si in range(1, n_seg - 1):
            L.append(f'        <body name="leaf{li}_seg{si}" pos="{seg_len:.6f} 0 0">')
            L.append(
                f'          <joint name="leaf{li}_seg{si}_joint" type="ball" '
                f'stiffness="{stiffs[si]:.6f}" damping="{damps[si]:.6f}"/>')
            L.append(
                f'          <geom name="leaf{li}_seg{si}_cap" type="capsule" fromto="0 0 0 {seg_len:.6f} 0 0" '
                f'size="{radii[si]:.6f}" contype="{ct.leaf[0]}" conaffinity="{ct.leaf[1]}" '
                f'rgba="{params.stem_rgba}" density="{cr.density:.1f}"/>')

        si = n_seg - 1
        L.append(f'        <body name="leaf{li}_seg{si}" pos="{seg_len:.6f} 0 0">')
        L.append(
            f'          <joint name="leaf{li}_seg{si}_joint" type="ball" '
            f'stiffness="{stiffs[si]:.6f}" damping="{damps[si]:.6f}"/>')
        L.append(
            f'          <geom name="leaf{li}_seg{si}_cap" type="capsule" fromto="0 0 0 {seg_len:.6f} 0 0" '
            f'size="{radii[si]:.6f}" contype="{ct.leaf[0]}" conaffinity="{ct.leaf[1]}" '
            f'rgba="{params.stem_rgba}" density="{cr.density:.1f}"/>')

        # ── trifoliate leaf (leaf-only petioles) ──
        if leaf.truss is None:
            lb = params.leaf_blade
            spread_rad = math.radians(lb.spread_angle)
            L.append(f'        <body name="leaf{li}_base" pos="{seg_len:.6f} 0 0">')
            L.append(f'          <joint name="leaf{li}_base_joint" type="ball" '
                      f'stiffness="{lb.base_joint_k:.6f}" damping="{lb.base_joint_d:.6f}"/>')
            # center leaflet
            L.append(
                f'          <geom name="leaf{li}_blade_c" type="box" '
                f'pos="{lb.length / 2:.6f} 0 0" '
                f'size="{lb.length / 2:.6f} {lb.width / 2:.6f} {lb.thickness / 2:.6f}" '
                f'contype="{ct.leaf[0]}" conaffinity="{ct.leaf[1]}" '
                f'rgba="{params.leaf_rgba}" density="{lb.density:.1f}"/>')
            # left leaflet
            q_l = q2s(quat_from_rotmat(_Rz(-spread_rad)))
            lx = lb.length / 2 * math.cos(spread_rad)
            ly = -lb.length / 2 * math.sin(spread_rad)
            L.append(
                f'          <geom name="leaf{li}_blade_l" type="box" '
                f'pos="{lx:.6f} {ly:.6f} 0" quat="{q_l}" '
                f'size="{lb.length / 2:.6f} {lb.width / 2:.6f} {lb.thickness / 2:.6f}" '
                f'contype="{ct.leaf[0]}" conaffinity="{ct.leaf[1]}" '
                f'rgba="{params.leaf_rgba}" density="{lb.density:.1f}"/>')
            # right leaflet
            q_r = q2s(quat_from_rotmat(_Rz(spread_rad)))
            rx = lb.length / 2 * math.cos(spread_rad)
            ry = lb.length / 2 * math.sin(spread_rad)
            L.append(
                f'          <geom name="leaf{li}_blade_r" type="box" '
                f'pos="{rx:.6f} {ry:.6f} 0" quat="{q_r}" '
                f'size="{lb.length / 2:.6f} {lb.width / 2:.6f} {lb.thickness / 2:.6f}" '
                f'contype="{ct.leaf[0]}" conaffinity="{ct.leaf[1]}" '
                f'rgba="{params.leaf_rgba}" density="{lb.density:.1f}"/>')
            L.append(f'        </body>')

        # ── truss stems (segmented) ──
        if leaf.truss is not None:
            tc = leaf.truss
            berry = params.berry
            stem_len = tc.stem_len
            n_stem = tc.stem_n_seg
            stem_seg_len = stem_len / n_stem
            stem_radii = [taper_radius(tc.stem_r_base, tc.stem_taper, i, n_stem) for i in range(n_stem)]
            stem_stiffs = [exp_stiffness(tc.stem_k_base, tc.stem_k_decay, i, n_stem) for i in range(n_stem)]
            stem_damps = [tc.stem_damping_ratio * k for k in stem_stiffs]
            # First stem segment
            L.append(f'          <body name="leaf{li}_stem_seg0" pos="{seg_len:.6f} 0 0">')
            L.append(f'          <joint name="leaf{li}_stem_seg0_joint" type="ball" stiffness="{stiffs[-1]:.6f}" damping="0.0"/>')
            L.append(
                f'          <geom name="leaf{li}_stem_seg0" type="capsule" fromto="0 0 0 {stem_seg_len:.6f} 0 0 " '
                f'size="{stem_radii[0]:.6f}" contype="{ct.structure[0]}" conaffinity="{ct.structure[1]}" '
                f'rgba="{params.stem_rgba}" density="{cr.density:.1f}"/>')

            # Middle stem segments
            for si in range(1, n_stem - 1):
                L.append(f'            <body name="leaf{li}_stem_seg{si}" pos="{stem_seg_len:.6f} 0 0">')
                L.append(
                    f'              <joint name="leaf{li}_stem_seg{si}_joint" type="ball" '
                    f'stiffness="{stem_stiffs[si]:.6f}" damping="{stem_damps[si]:.6f}"/>')
                L.append(
                    f'              <geom name="leaf{li}_stem_seg{si}" type="capsule" fromto="0 0 0 {stem_seg_len:.6f} 0 0 " '
                    f'size="{stem_radii[si]:.6f}" contype="{ct.structure[0]}" conaffinity="{ct.structure[1]}" '
                    f'rgba="{params.stem_rgba}" density="{cr.density:.1f}"/>')

            # Last stem segment — branches attach here
            si = n_stem - 1
            L.append(f'            <body name="leaf{li}_stem_seg{si}" pos="{stem_seg_len:.6f} 0 0">')
            L.append(
                f'              <joint name="leaf{li}_stem_seg{si}_joint" type="ball" '
                f'stiffness="{stem_stiffs[si]:.6f}" damping="{stem_damps[si]:.6f}"/>')
            L.append(
                f'              <geom name="leaf{li}_stem_seg{si}" type="capsule" fromto="0 0 0 {stem_seg_len:.6f} 0 0 " '
                f'size="{stem_radii[si]:.6f}" contype="{ct.structure[0]}" conaffinity="{ct.structure[1]}" '
                f'rgba="{params.stem_rgba}" density="{cr.density:.1f}"/>')

            PED_R = berry.ped_r
            PED_K = berry.ped_k
            PED_D = berry.ped_d

            # Branches — all pedicels lie in a single vertical plane at plane_az
            plane_az = math.radians(tc.plane_az_deg)
            bi = 0
            for br in tc.branches:
                droop_rad = math.radians(br.droop_deg)
                # For simplicity, attach all branches to last segment with local offset
                local_z_in_seg = br.frac * stem_len - (n_stem - 1) * stem_seg_len

                dx = 0
                dy = br.ped_len * math.cos(droop_rad) * math.cos(plane_az)
                dz = -math.sin(droop_rad) * br.ped_len

                L.append(f'              <body name="leaf{li}_branch{bi}" pos="{local_z_in_seg:.6f} 0 0">')
                L.append(
                    f'                <joint name="leaf{li}_branch{bi}_joint" type="ball" '
                    f'stiffness="{PED_K:.6f}" damping="{PED_D:.6f}"/>')
                L.append(
                    f'                <geom name="leaf{li}_branch{bi}_stem" type="capsule" fromto="0 0 0 {dx:.6f} {dy:.6f} {dz:.6f}" '
                    f'size="{PED_R:.6f}" contype="{ct.structure[0]}" conaffinity="{ct.structure[1]}" '
                    f'rgba="{params.stem_rgba}" density="{cr.density:.1f}"/>')

                if br.has_sub:
                    sub_droop_rad = math.radians(br.sub_droop_deg)
                    sdx = 0
                    sdy = br.sub_len * math.cos(sub_droop_rad) * math.cos(plane_az)
                    sdz = -math.sin(sub_droop_rad) * br.sub_len
                    L.append(f'                <body name="leaf{li}_branch{bi}b" pos="{dx:.6f} {dy:.6f} {dz:.6f}">')
                    L.append(
                        f'                  <joint name="leaf{li}_branch{bi}b_joint" type="ball" '
                        f'stiffness="{PED_K * 0.7:.6f}" damping="{PED_D * 0.7:.6f}"/>')
                    L.append(
                        f'                  <geom name="leaf{li}_branch{bi}b_stem" type="capsule" fromto="0 0 0 {sdx:.6f} {sdy:.6f} {sdz:.6f}" '
                        f'size="{PED_R:.6f}" contype="{ct.structure[0]}" conaffinity="{ct.structure[1]}" '
                        f'rgba="{params.stem_rgba}" density="{cr.density:.1f}"/>')
                    # berry on its own body
                    L.append(f'                  <body name="leaf{li}_berry{bi}b" pos="{sdx:.6f} {sdy:.6f} {sdz:.6f}">')
                    L.append(
                        f'                    <joint name="leaf{li}_berry{bi}b_joint" type="ball" '
                        f'stiffness="{berry.joint_k:.6f}" damping="{berry.joint_d:.6f}"/>')
                    L.extend(berry_geoms(f"leaf{li}_berry{bi}b_geom", br.sub_berry_r_scale,
                                         "                    ", (sdx, sdy, sdz)))
                    L.append(f'                  </body>')
                    L.append(f'                </body>')
                else:
                    # berry on its own body at end of pedicel
                    L.append(f'                <body name="leaf{li}_berry{bi}" pos="{dx:.6f} {dy:.6f} {dz:.6f}">')
                    L.append(
                        f'                  <joint name="leaf{li}_berry{bi}_joint" type="ball" '
                        f'stiffness="{berry.joint_k:.6f}" damping="{berry.joint_d:.6f}"/>')
                    L.extend(berry_geoms(f"leaf{li}_berry{bi}_geom", br.berry_r_scale,
                                         "                  ", (dx, dy, dz)))
                    L.append(f'                </body>')

                L.append(f'              </body>')
                bi += 1

            # Tip berry at end of last stem segment
            L.append(f'              <body name="leaf{li}_ped_tip" pos="{stem_seg_len:.6f} 0 0">')
            L.append(
                f'                <joint name="leaf{li}_ped_tip_joint" type="ball" '
                f'stiffness="{PED_K:.6f}" damping="{PED_D:.6f}"/>')
            L.append(
                f'                <geom name="leaf{li}_ped_tip_cap" type="capsule" fromto="0 0 0 {tc.tip_ped_len:.6f} 0 0" '
                f'size="{PED_R:.6f}" contype="{ct.structure[0]}" conaffinity="{ct.structure[1]}" '
                f'rgba="{params.stem_rgba}" density="{cr.density:.1f}"/>')
            L.append(f'                <body name="leaf{li}_berry_tip" pos="{tc.tip_ped_len:.6f} 0 0">')
            L.append(
                f'                  <joint name="leaf{li}_berry_tip_joint" type="ball" '
                f'stiffness="{berry.joint_k:.6f}" damping="{berry.joint_d:.6f}"/>')
            L.extend(berry_geoms(f"leaf{li}_berry_tip_geom", tc.tip_berry_r_scale,
                                 "                  ", (tc.tip_ped_len, 0.0, 0.0)))
            L.append(f'                </body>')
            L.append(f'              </body>')

            # Close last stem segment
            L.append(f'            </body>')  # stem_seg last
            for si in range(n_stem - 2, 0, -1):
                L.append(f'            </body>')
            L.append(f'          </body>')  # stem_seg0

        L.append('        </body>')  # last petiole seg
        for si in range(n_seg - 2, 0, -1):
            L.append('        </body>')
        L.append('      </body>')  # petiole seg0

    L.append('    </body>')  # crown
    L.append('  </worldbody>')
    L.append('</mujoco>')

    if asset_lines:
        L[asset_insert_index:asset_insert_index] = ['  <asset>', *asset_lines, '  </asset>']

    return '\n'.join(L)

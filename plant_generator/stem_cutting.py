"""Contact-triggered cutting and gripper release for composed picker scenes."""
from __future__ import annotations

import logging
import re

import mujoco
import numpy as np

from .finger_hitboxes import HITBOX_NAMES


_STEM_NAME = re.compile(r"^leaf(\d+)_stem_seg(\d+)$")
logger = logging.getLogger(__name__)


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


class StemCutter:
    """Cut after both blade hitboxes touch one segment; release on gripper open."""

    def __init__(self, model: mujoco.MjModel, contact_steps: int = 3) -> None:
        if contact_steps < 1:
            raise ValueError("contact_steps must be at least one")
        self.model = model
        self.contact_steps = contact_steps
        self._hitbox_side = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name): side
            for side, name in enumerate(HITBOX_NAMES)
        }
        self._hitbox_side.pop(-1, None)
        self._stem_geoms: dict[int, tuple[int, int]] = {}
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            match = _STEM_NAME.match(name)
            if match:
                self._stem_geoms[geom_id] = tuple(map(int, match.groups()))

        self._cut_links: dict[tuple[int, int], int] = {}
        self._hold_welds: dict[tuple[int, int], int] = {}
        self._stem_bodies: dict[tuple[int, int], int] = {}
        self._stem_joints: dict[tuple[int, int], int] = {}
        self._left_finger = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        for leaf, segment in set(self._stem_geoms.values()):
            key = (leaf, segment)
            self._stem_bodies[key] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"leaf{leaf}_stem_seg{segment}")
            self._stem_joints[key] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"leaf{leaf}_stem_seg{segment}_joint")
            self._cut_links[key] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, f"cut_leaf{leaf}_stem_seg{segment}")
            self._hold_welds[key] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, f"hold_leaf{leaf}_stem_seg{segment}")
        self._cut_links = {key: value for key, value in self._cut_links.items() if value >= 0}
        self._hold_welds = {key: value for key, value in self._hold_welds.items() if value >= 0}
        self._contact_steps: dict[tuple[int, int], int] = {}
        self._cut_segments: set[tuple[int, int]] = set()
        self._active_holds: set[tuple[int, int]] = set()
        if not self._hitbox_side or not self._cut_links:
            logger.warning("Stem cutting constraints missing; regenerate the composed scene")

    def _release_holds(self, data: mujoco.MjData) -> None:
        for key in tuple(self._active_holds):
            weld_id = self._hold_welds.get(key, -1)
            if weld_id >= 0:
                data.eq_active[weld_id] = 0
                logger.info("Released detached stem fragment %s from gripper", self._body_name(key))
        self._active_holds.clear()

    def _body_name(self, key: tuple[int, int]) -> str:
        return f"leaf{key[0]}_stem_seg{key[1]}"

    def _set_hold_pose(self, data: mujoco.MjData, key: tuple[int, int], weld_id: int) -> None:
        """Set body2's weld reference pose relative to body1 at the cut instant."""
        finger, stem = self._left_finger, self._stem_bodies[key]
        if finger < 0 or stem < 0:
            raise RuntimeError("Hold weld references a missing body")
        rotation = np.asarray(data.xmat[finger]).reshape(3, 3)
        relpos = rotation.T @ (data.xpos[stem] - data.xpos[finger])
        relquat = _quat_multiply(_quat_conjugate(data.xquat[finger]), data.xquat[stem])
        eq_data = self.model.eq_data[weld_id]
        if len(eq_data) < 10:
            raise RuntimeError("MuJoCo weld equality data has an unexpected layout")
        eq_data[3:6] = relpos
        eq_data[6:10] = relquat

    def _cut(self, data: mujoco.MjData, key: tuple[int, int], contact_pos: np.ndarray,
             *, releasing: bool) -> str:
        leaf, _segment = key
        candidates = [candidate for candidate in self._cut_links
                      if candidate[0] == leaf and bool(data.eq_active[self._cut_links[candidate]])
                      and data.xpos[self._stem_bodies[candidate], 1] > contact_pos[1]]
        if not candidates:
            logger.warning("No uncut +Y stem segment found above contact on leaf%d", leaf)
            self._cut_segments.add(key)
            return self._body_name(key)

        cut_key = min(candidates, key=lambda candidate: float(np.sum(
            (data.xpos[self._stem_bodies[candidate]] - contact_pos) ** 2)))
        link_id = self._cut_links[cut_key]
        weld_id = self._hold_welds.get(cut_key, -1)
        data.eq_active[link_id] = 0

        # The ball spring at the severed boundary must not tether the fragment
        # rotationally after its translational anchor has been released.
        joint_id = self._stem_joints[cut_key]
        if joint_id >= 0:
            self.model.jnt_stiffness[joint_id] = 0.0
            dof_adr = int(self.model.jnt_dofadr[joint_id])
            following = self.model.jnt_dofadr[self.model.jnt_dofadr > dof_adr]
            dof_count = int((following.min() if len(following) else self.model.nv) - dof_adr)
            self.model.dof_damping[dof_adr:dof_adr + dof_count] = 0.0

        self._cut_segments.add(cut_key)
        if weld_id >= 0 and not releasing:
            self._set_hold_pose(data, cut_key, weld_id)
            data.eq_active[weld_id] = 1
            self._active_holds.add(cut_key)
            action = "detached and held by left finger"
        else:
            action = "detached without hold (gripper is opening)"
        logger.info("Stem cut: %s, contact=(%.4f, %.4f, %.4f) m; %s",
                    self._body_name(cut_key), *contact_pos, action)
        return self._body_name(cut_key)

    def step(self, data: mujoco.MjData, *, releasing: bool = False) -> str | None:
        """Update contact debouncing and apply any cut; cutting is state agnostic."""
        if releasing:
            self._release_holds(data)

        contacts: dict[tuple[int, int], dict[int, list[np.ndarray]]] = {}
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            side = self._hitbox_side.get(geom1)
            stem_geom = geom2
            if side is None:
                side = self._hitbox_side.get(geom2)
                stem_geom = geom1
            if side is None or stem_geom not in self._stem_geoms:
                continue
            key = self._stem_geoms[stem_geom]
            contacts.setdefault(key, {}).setdefault(side, []).append(
                np.asarray(contact.pos, dtype=float).copy())

        triggered = {
            key: sides for key, sides in contacts.items()
            if set(sides) == {0, 1} and key not in self._cut_segments
        }
        for key in tuple(self._contact_steps):
            if key not in triggered:
                self._contact_steps.pop(key, None)
        for key, sides in triggered.items():
            self._contact_steps[key] = self._contact_steps.get(key, 0) + 1
            if self._contact_steps[key] < self.contact_steps:
                continue
            points = [point for side_points in sides.values() for point in side_points]
            contact_pos = np.mean(points, axis=0)
            return self._cut(data, key, contact_pos, releasing=releasing)
        return None

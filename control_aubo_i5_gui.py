"""Control the AUBO i5 in a MuJoCo scene using six live joint sliders.

Run after creating ``tmp/strawberry_picker.xml``::

    python control_aubo_i5_gui.py
    python control_aubo_i5_gui.py --scene path/to/scene.xml
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import ttk

import mujoco
import mujoco.viewer
import numpy as np
from plant_generator.stem_cutting import StemCutter
from plant_generator.finger_hitboxes import HALF_SIZE, LOCAL_CENTERS

ACTUATORS = (
    ("Shoulder pan", "shoulder_pan", "shoulder_pan_joint", -3.04, 3.04),
    ("Shoulder lift", "shoulder_lift", "shoulder_lift_joint", -3.04, 3.04),
    ("Elbow", "elbow", "elbow_joint", -3.1415, 3.1415),
    ("Wrist 1", "wrist_1", "wrist_1_joint", -3.04, 3.04),
    ("Wrist 2", "wrist_2", "wrist_2_joint", -3.04, 3.04),
    ("Wrist 3", "wrist_3", "wrist_3_joint", -3.04, 3.04),
)
HOME = (0.0, -1.2, 1.2, 0.0, 0.0, 0.0)
GRIPPER_HOME = 0.47
RAD_TO_DEG = 180.0 / 3.141592653589793


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"Scene is missing AUBO i5 item {name!r}")
    return value


def _update_finger_side_planes(
    scene: mujoco.MjvScene,
    data: mujoco.MjData,
    finger_body_ids: tuple[int, int],
) -> None:
    """Draw translucent overlays over the AG95 inner finger contact planes."""
    scene.ngeom = len(finger_body_ids)
    # Local X is the thin length, Y is width, and Z is height.
    half_size = HALF_SIZE
    colors = (
        np.array([0.12, 0.75, 1.0, 0.32], dtype=np.float32),
        np.array([1.0, 0.38, 0.12, 0.32], dtype=np.float32),
    )
    for plane_index, body_id in enumerate(finger_body_ids):
        rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
        position = data.xpos[body_id] + rotation @ LOCAL_CENTERS[plane_index]
        mujoco.mjv_initGeom(
            scene.geoms[plane_index], mujoco.mjtGeom.mjGEOM_BOX,
            half_size, position, rotation.reshape(-1), colors[plane_index],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("tmp/strawberry_picker.xml"),
                        help="generated MuJoCo scene XML")
    parser.add_argument("--sim-speed", type=float, default=1.0, metavar="FACTOR",
                        help="simulation time multiplier relative to wall time (default: 1)")
    parser.add_argument("--render-hz", type=float, default=60.0, metavar="HZ",
                        help="maximum MuJoCo viewer refresh rate (default: 60)")
    parser.add_argument("--ui-hz", type=float, default=30.0, metavar="HZ",
                        help="joint readout and slider update rate (default: 30)")
    parser.add_argument("--gripper-rate", type=float, default=0.25, metavar="RAD_PER_SEC",
                        help="maximum AG95 target change per simulated second (default: 0.25)")
    parser.add_argument("--joint-rate", type=float, default=0.8, metavar="RAD_PER_SEC",
                        help="maximum arm joint target speed (default: 0.8)")
    parser.add_argument("--joint-accel", type=float, default=2.0, metavar="RAD_PER_SEC2",
                        help="maximum arm joint target acceleration (default: 2.0)")
    parser.add_argument("--log-file", type=Path, default=Path("tmp/stem_cutting.log"),
                        help="file for controller and stem-cut logs (default: tmp/stem_cutting.log)")
    parser.add_argument("--debug", action="store_true",
                        help="show translucent planes over the gripper finger contact sides")
    parser.add_argument("--disable-stem-cutting", action="store_true",
                        help="disable runtime stem cuts and fragment hold/release updates")
    args = parser.parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(args.log_file, encoding="utf-8")],
        force=True,
    )
    logging.getLogger(__name__).info("Logging to console and %s", args.log_file.resolve())
    if not 0.25 <= args.sim_speed <= 8.0:
        parser.error("--sim-speed must be between 0.25 and 8")
    if (args.render_hz <= 0 or args.ui_hz <= 0 or args.gripper_rate <= 0
            or args.joint_rate <= 0 or args.joint_accel <= 0):
        parser.error("render/UI rates and joint/gripper limits must be positive")
    if not args.scene.is_file():
        parser.error(f"Scene not found: {args.scene}. Generate it with create_picker_scene.py first.")

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    actuator_ids = [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, row[1]) for row in ACTUATORS]
    actuator_ids.append(_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator"))
    # Apply the arm damping here as well, so scenes generated before the gain
    # adjustment get the same quieter response when loaded by this controller.
    for actuator_id in actuator_ids[:6]:
        model.actuator_biasprm[actuator_id, 2] = -50.0
    qpos_addresses = [int(model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, row[2])])
                      for row in ACTUATORS]
    qpos_addresses.append(int(model.jnt_qposadr[
        _id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_outer_knuckle_joint")]))
    right_gripper_qpos = int(model.jnt_qposadr[
        _id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_outer_knuckle_joint")])
    data.ctrl[actuator_ids] = [*HOME, GRIPPER_HOME]
    data.qpos[qpos_addresses[:6]] = HOME
    data.qpos[qpos_addresses[-1]] = GRIPPER_HOME
    data.qpos[right_gripper_qpos] = GRIPPER_HOME
    mujoco.mj_forward(model, data)
    stem_cutter = None if args.disable_stem_cutting else StemCutter(model)
    if args.disable_stem_cutting:
        logging.getLogger(__name__).info(
            "Stem cutting disabled; hitbox contacts and scene constraints remain active")

    window = tk.Tk()
    window.title("AUBO i5 joint control")
    window.resizable(False, False)
    frame = ttk.Frame(window, padding=12)
    frame.grid(sticky="nsew")
    ttk.Label(frame, text="Joint target (degrees)", font=("Segoe UI", 10, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

    target_vars: list[tk.DoubleVar] = []
    current_labels: list[ttk.Label] = []
    for row_num, (label, _actuator, _joint, low, high), home in zip(range(1, 7), ACTUATORS, HOME):
        ttk.Label(frame, text=label, width=15).grid(row=row_num, column=0, sticky="w")
        variable = tk.DoubleVar(value=home * RAD_TO_DEG)
        target_vars.append(variable)
        ttk.Scale(frame, from_=low * RAD_TO_DEG, to=high * RAD_TO_DEG,
                  variable=variable, orient="horizontal", length=300).grid(
                      row=row_num, column=1, padx=8, sticky="ew")
        current = ttk.Label(frame, text="   0.0°", width=8, anchor="e")
        current.grid(row=row_num, column=2, sticky="e")
        current_labels.append(current)

    ttk.Label(frame, text="AG95 gripper", width=15).grid(row=7, column=0, sticky="w")
    gripper_target = tk.DoubleVar(value=GRIPPER_HOME)
    ttk.Scale(frame, from_=0.0, to=0.943, variable=gripper_target,
              orient="horizontal", length=300).grid(row=7, column=1, padx=8, sticky="ew")
    gripper_current = ttk.Label(frame, text="  0.00", width=8, anchor="e")
    gripper_current.grid(row=7, column=2, sticky="e")
    current_labels.append(gripper_current)

    pending_targets = [*HOME, GRIPPER_HOME]
    commanded_joints = list(HOME)
    commanded_joint_velocities = [0.0] * len(HOME)
    commanded_gripper = [GRIPPER_HOME]
    active_speed = [args.sim_speed]
    shared_lock = threading.Lock()
    closing = threading.Event()
    viewer_errors: list[BaseException] = []

    def simulation() -> None:
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
                # Show collision meshes without overlapping visual robot meshes.
                viewer.opt.geomgroup[2] = 0
                viewer.opt.geomgroup[3] = 1
                finger_body_ids: tuple[int, int] | None = None
                if args.debug:
                    finger_body_ids = (
                        _id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
                        _id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
                    )
                    if viewer.user_scn is None:
                        raise RuntimeError("MuJoCo viewer has no user scene for debug overlays")
                    _update_finger_side_planes(viewer.user_scn, data, finger_body_ids)
                    logging.getLogger(__name__).info("Debug finger-side planes enabled")
                previous = time.perf_counter()
                next_render = previous
                accumulated = 0.0
                render_period = 1.0 / args.render_hz
                # Bound catch-up work so a slow render does not cause a growing
                # backlog that makes controls and the viewer feel frozen.
                max_steps_per_frame = 200
                max_accumulated = max_steps_per_frame * model.opt.timestep
                while viewer.is_running() and not closing.is_set():
                    now = time.perf_counter()
                    wall_delta = min(now - previous, 0.25)
                    previous = now
                    with shared_lock, viewer.lock():
                        accumulated = min(accumulated + wall_delta * active_speed[0], max_accumulated)
                        steps = min(int(accumulated / model.opt.timestep), max_steps_per_frame)
                        # Advance several physics steps per rendered frame. This
                        # avoids paying viewer rendering cost at every 2 ms step.
                        for _ in range(steps):
                            timestep = model.opt.timestep
                            for joint_index, target in enumerate(pending_targets[:6]):
                                error = target - commanded_joints[joint_index]
                                if abs(error) < 1e-9:
                                    commanded_joints[joint_index] = target
                                    commanded_joint_velocities[joint_index] = 0.0
                                    continue
                                desired_speed = math.copysign(
                                    min(args.joint_rate,
                                        math.sqrt(2 * args.joint_accel * abs(error))),
                                    error,
                                )
                                speed_change = args.joint_accel * timestep
                                current_speed = commanded_joint_velocities[joint_index]
                                current_speed += max(-speed_change, min(speed_change,
                                    desired_speed - current_speed))
                                step = current_speed * timestep
                                if abs(step) >= abs(error) and step * error > 0:
                                    commanded_joints[joint_index] = target
                                    current_speed = 0.0
                                else:
                                    commanded_joints[joint_index] += step
                                commanded_joint_velocities[joint_index] = current_speed

                            target = pending_targets[-1]
                            was_opening = target > commanded_gripper[0] + 1e-9
                            max_delta = args.gripper_rate * timestep
                            commanded_gripper[0] += max(-max_delta, min(max_delta,
                                target - commanded_gripper[0]))
                            data.ctrl[actuator_ids] = [*commanded_joints, commanded_gripper[0]]
                            mujoco.mj_step(model, data)
                            if stem_cutter is not None:
                                stem_cutter.step(data, releasing=was_opening)
                        accumulated -= steps * model.opt.timestep
                        if args.debug and finger_body_ids is not None:
                            _update_finger_side_planes(viewer.user_scn, data, finger_body_ids)
                    viewer.sync()
                    next_render += render_period
                    now = time.perf_counter()
                    if next_render < now:
                        next_render = now + render_period
                    time.sleep(max(0.0, next_render - now))
        except BaseException as exc:
            viewer_errors.append(exc)
            closing.set()

    sim_thread = threading.Thread(target=simulation, name="mujoco-viewer", daemon=True)

    def close() -> None:
        closing.set()
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", close)

    def set_home() -> None:
        for variable, angle in zip(target_vars, HOME):
            variable.set(angle * RAD_TO_DEG)
        gripper_target.set(GRIPPER_HOME)

    buttons = ttk.Frame(frame)
    buttons.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(10, 0))
    ttk.Button(buttons, text="Home", command=set_home).pack(side="left")
    ttk.Label(buttons, text="Target · current angle").pack(side="left", padx=10)
    speed_var = tk.DoubleVar(value=args.sim_speed)
    speed_label = ttk.Label(buttons, text=f"Speed: {args.sim_speed:g}×")
    speed_label.pack(side="right", padx=(8, 0))
    ttk.Scale(buttons, from_=0.25, to=8.0, variable=speed_var,
              orient="horizontal", length=140).pack(side="right")

    def update_ui() -> None:
        if closing.is_set():
            if viewer_errors:
                window.destroy()
                raise RuntimeError(f"MuJoCo viewer failed: {viewer_errors[0]}") from viewer_errors[0]
            return
        with shared_lock:
            pending_targets[:] = [var.get() / RAD_TO_DEG for var in target_vars] + [gripper_target.get()]
            active_speed[0] = max(0.25, speed_var.get())
            current_angles = [float(data.qpos[address]) for address in qpos_addresses]
        for label, angle in zip(current_labels[:6], current_angles[:6]):
            label.configure(text=f"{angle:7.1f}°")
        gripper_current.configure(text=f"{current_angles[6]:6.3f}")
        speed_label.configure(text=f"Speed: {active_speed[0]:.1f}×")
        window.after(max(1, round(1000 / args.ui_hz)), update_ui)

    sim_thread.start()
    try:
        update_ui()
        window.mainloop()
    finally:
        closing.set()
        sim_thread.join(timeout=2)


if __name__ == "__main__":
    main()

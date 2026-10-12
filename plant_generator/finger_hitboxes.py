"""Shared local geometry for the AG95 finger-side blade hitboxes."""
from __future__ import annotations

import numpy as np


# The AG95 right finger frame is rotated 180 degrees around local Z relative
# to the left finger frame. Reverse local Y on the left so both boxes occupy
# the same gripper-relative height.
_LOCAL_X = -0.0286
_LOCAL_Y = 0.00875 #+ 0.01
_LOCAL_Z = 0.032
LOCAL_CENTERS = (
    np.array([_LOCAL_X, _LOCAL_Y-0.01, _LOCAL_Z], dtype=float),  # left / blue
    np.array([_LOCAL_X, _LOCAL_Y+0.01, _LOCAL_Z], dtype=float),   # right / orange
)
HALF_SIZE = np.array([0.001, 0.010 / 7, 0.013], dtype=float)
HITBOX_NAMES = ("left_finger_cut_hitbox", "right_finger_cut_hitbox")

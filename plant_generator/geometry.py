"""Pure geometry and math helpers — no dependencies on params."""

import math
import numpy as np


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
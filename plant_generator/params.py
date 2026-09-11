"""Dataclasses for plant configuration parameters."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from .geometry import exp_stiffness, logistic_stiffness

# ── type alias ──────────────────────────────────────────────────
StiffnessFunc = Callable[[int, int], float]
# (segment_index, total_segments) -> stiffness (N·m/rad)


# ── petiole profiles ────────────────────────────────────────────
@dataclass
class PetioleProfile:
    n_seg: int = 12
    length: float = 0.18
    r_base: float = 0.0015
    taper: float = 0.6
    stiffness: StiffnessFunc = field(default_factory=lambda: lambda i, n: exp_stiffness(1.0, 2.0, i, n) / 2)
    damping_ratio: float = 0.4


def exp_profile(k_base: float = 1.0, decay: float = 2.0, **overrides) -> PetioleProfile:
    """Exponential decay stiffness — used for berry petioles."""
    return PetioleProfile(stiffness=lambda i, n: exp_stiffness(k_base, decay, i, n) / 2, **overrides)


def logistic_profile(k_base: float = 2.5, alpha: float = 9.0, i_break: float = 6.5,
                     k_min: float = 0.03, **overrides) -> PetioleProfile:
    """Logistic stiffness — used for leaf petioles."""
    return PetioleProfile(stiffness=lambda i, n: logistic_stiffness(k_base, alpha, i_break, k_min, i, n) / 2, **overrides)


# ── leaf blade ─────────────────────────────────────────────────
@dataclass
class LeafBlade:
    length: float = 0.095
    width: float = 0.040
    thickness: float = 0.003
    spread_angle: float = 40.0      # degrees
    density: float = 900.0
    base_joint_k: float = 0.2
    base_joint_d: float = 0.001


# ── branch spec ────────────────────────────────────────────────
@dataclass
class BranchSpec:
    frac: float                      # fraction along stem where branch attaches
    ped_len: float                   # pedicel length (m)
    droop_deg: float                 # droop angle (degrees)
    berry_r_scale: float = 1.0
    has_sub: bool = False
    sub_len: float = 0.0
    sub_droop_deg: float = 0.0
    sub_berry_r_scale: float = 1.0


# ── truss config ────────────────────────────────────────────────
@dataclass
class TrussConfig:
    stem_len: float = 0.07
    stem_n_seg: int = 3
    stem_r_base: float = 0.001
    stem_taper: float = 0.5
    stem_k_base: float = 0.15
    stem_k_decay: float = 2.5
    stem_damping_ratio: float = 0.4
    plane_az_deg: float = 0.0       # azimuth of the pedicel plane (degrees)
    branches: list[BranchSpec] = field(default_factory=list)
    tip_berry_r_scale: float = 1.0
    tip_ped_len: float = 0.025


# ── crown ───────────────────────────────────────────────────────
@dataclass
class CrownConfig:
    radius: float = 0.02
    half_height: float = 0.01
    density: float = 450.0


# ── berry mesh ──────────────────────────────────────────────────
@dataclass
class BerryMeshConfig:
    """Parameters for procedural strawberry mesh generation."""
    n_lat: int = 24                  # latitude rings (mesh resolution along height)
    n_lon: int = 32                  # longitude slices (mesh resolution around circumference)
    max_radius_frac: float = 0.55   # max radius as fraction of height (unit scale)
    belly_pos: float = 0.42         # height fraction where berry is widest (0=calyx, 1=tip)
    belly_width: float = 1.0         # radius at belly as fraction of max_radius_frac
    tip_round: float = 0.15         # tip rounding (0=sharp cone, 1=hemispherical)
    neck_depth: float = 0.12        # neck constriction depth below calyx (fraction of max radius)
    neck_pos: float = 0.12          # height fraction where neck is narrowest
    calyx_flare: float = 0.40       # calyx flare width as fraction of max_radius_frac
    calyx_height: float = 0.06      # calyx region height as fraction of total height
    bump_amp: float = 0.025         # seed bump amplitude (fraction of local radius)
    bump_rings: int = 8             # approximate number of seed rows along height
    bump_spirals: int = 5           # number of seed spiral columns around circumference


# ── berry ───────────────────────────────────────────────────────
@dataclass
class BerryConfig:
    r_base: float = 0.012
    density: float = 1450.0
    oblateness: float = 1.15
    joint_k: float = 0.008
    joint_d: float = 0.004
    ped_r: float = 0.0007
    ped_k: float = 0.008
    ped_d: float = 0.003


# ── collision groups ────────────────────────────────────────────
@dataclass
class CollisionGroups:
    ground: tuple[int, int] = (1, 15)
    structure: tuple[int, int] = (2, 25)
    leaf: tuple[int, int] = (4, 25)
    berry: tuple[int, int] = (8, 15)


# ── leaf entry ──────────────────────────────────────────────────
@dataclass
class LeafEntry:
    azimuth_deg: float              # degrees from +X
    elevation_deg: float            # degrees from horizontal
    truss: TrussConfig | None = None   # None = leaf-only petiole


# ── top-level params ────────────────────────────────────────────
@dataclass
class PlantParams:
    crown: CrownConfig = field(default_factory=CrownConfig)
    berry_profile: PetioleProfile = field(default_factory=exp_profile)
    leaf_profile: PetioleProfile = field(default_factory=logistic_profile)
    leaf_blade: LeafBlade = field(default_factory=LeafBlade)
    berry: BerryConfig = field(default_factory=BerryConfig)
    collision: CollisionGroups = field(default_factory=CollisionGroups)
    leaves: list[LeafEntry] = field(default_factory=lambda: [
        LeafEntry(0, 55, truss=TrussConfig(stem_len=0.070, branches=[
            BranchSpec(0.45, 0.035, 40), BranchSpec(0.70, 0.030, 40)], tip_ped_len=0.025)),
        LeafEntry(137, 45, truss=TrussConfig(stem_len=0.060, branches=[
            BranchSpec(0.40, 0.032, 25),
            BranchSpec(0.70, 0.028, 20, 0.9, True, sub_len=0.022, sub_droop_deg=35, sub_berry_r_scale=0.81),
            BranchSpec(0.85, 0.025, 30, 0.85)], tip_ped_len=0.020)),
        LeafEntry(274, 35, truss=TrussConfig(stem_len=0.055, branches=[
            BranchSpec(0.45, 0.025, 18, 1.0, True, sub_len=0.018, sub_droop_deg=30, sub_berry_r_scale=0.85),
            BranchSpec(0.75, 0.022, 28)], tip_ped_len=0.018)),
        LeafEntry(90, 60, truss=TrussConfig(stem_len=0.065, branches=[
            BranchSpec(0.40, 0.030, 35), BranchSpec(0.65, 0.025, 35)], tip_ped_len=0.020)),
        LeafEntry(200, 70),
        LeafEntry(310, 80),
        LeafEntry(45, 75),
    ])
    stem_rgba: str = "0.38 0.40 0.20 1"
    leaf_rgba: str = "0.25 0.55 0.22 1"
    berry_rgba: str = "0.85 0.1 0.08 1"
    model_name: str = "crown_petioles_truss"
    timestep: float = 0.002
    integrator: str = "implicitfast"
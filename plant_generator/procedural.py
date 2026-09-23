"""Sample varied plants from the original crown model's parameters."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .params import (
    BranchSpec,
    LeafEntry,
    PlantParams,
    TrussConfig,
    exp_profile,
    logistic_profile,
)


@dataclass(frozen=True)
class FloatRange:
    low: float
    high: float

    def sample(self, rng: random.Random) -> float:
        if self.low > self.high:
            raise ValueError(f"Invalid range: {self.low} > {self.high}")
        return rng.uniform(self.low, self.high)


@dataclass(frozen=True)
class PlantRanges:
    """Editable ranges in SI units; angles are degrees above the ground plane."""

    fruit_stems: tuple[int, int] = (3, 5)
    leaf_stems: tuple[int, int] = (3, 6)
    azimuth_jitter_fraction: float = 0.25
    fruit_elevation: FloatRange = field(default_factory=lambda: FloatRange(20, 45))
    leaf_elevation: FloatRange = field(default_factory=lambda: FloatRange(55, 80))
    crown_radius: FloatRange = field(default_factory=lambda: FloatRange(0.017, 0.024))
    crown_half_height: FloatRange = field(default_factory=lambda: FloatRange(0.008, 0.012))
    fruit_petiole_length: FloatRange = field(default_factory=lambda: FloatRange(0.15, 0.21))
    leaf_petiole_length: FloatRange = field(default_factory=lambda: FloatRange(0.15, 0.22))
    petiole_radius_scale: FloatRange = field(default_factory=lambda: FloatRange(0.85, 1.15))
    petiole_taper: FloatRange = field(default_factory=lambda: FloatRange(0.5, 0.72))
    fruit_stiffness: FloatRange = field(default_factory=lambda: FloatRange(0.8, 1.2))
    leaf_stiffness: FloatRange = field(default_factory=lambda: FloatRange(2.0, 3.0))
    damping_ratio: FloatRange = field(default_factory=lambda: FloatRange(0.32, 0.48))
    leaf_length: FloatRange = field(default_factory=lambda: FloatRange(0.080, 0.110))
    leaf_width: FloatRange = field(default_factory=lambda: FloatRange(0.034, 0.048))
    leaflet_spread: FloatRange = field(default_factory=lambda: FloatRange(32, 48))
    berry_radius: FloatRange = field(default_factory=lambda: FloatRange(0.010, 0.014))
    stem_length: FloatRange = field(default_factory=lambda: FloatRange(0.05, 0.08))
    stem_radius: FloatRange = field(default_factory=lambda: FloatRange(0.00085, 0.00115))
    stem_stiffness: FloatRange = field(default_factory=lambda: FloatRange(0.12, 0.19))
    branches_per_truss: tuple[int, int] = (1, 3)
    branch_fraction: FloatRange = field(default_factory=lambda: FloatRange(0.43, 0.82))
    pedicel_length: FloatRange = field(default_factory=lambda: FloatRange(0.021, 0.038))
    pedicel_droop: FloatRange = field(default_factory=lambda: FloatRange(18, 45))
    berry_scale: FloatRange = field(default_factory=lambda: FloatRange(0.8, 1.12))
    sub_branch_probability: float = 0.18


def _stem_azimuths(count: int, rng: random.Random, jitter_fraction: float) -> list[float]:
    """One jittered angle per equal circular sector prevents large bare wedges."""
    if not 0 <= jitter_fraction < 0.5:
        raise ValueError("azimuth_jitter_fraction must be in [0, 0.5)")
    step = 360 / count
    offset = rng.uniform(0, 360)
    return [(offset + i * step + rng.uniform(-jitter_fraction, jitter_fraction) * step) % 360
            for i in range(count)]


def sample_plant(rng: random.Random, ranges: PlantRanges = PlantRanges()) -> PlantParams:
    """Generate a plant with balanced azimuths and independent size/shape variation."""
    params = PlantParams()
    fruit_count = rng.randint(*ranges.fruit_stems)
    leaf_count = rng.randint(*ranges.leaf_stems)
    if fruit_count < 1 or leaf_count < 1:
        raise ValueError("A generated plant needs at least one fruit and one leaf stem")
    total = fruit_count + leaf_count
    azimuths = _stem_azimuths(total, rng, ranges.azimuth_jitter_fraction)
    # Spread fruit stems across the same circle; the remaining sectors get leaves.
    start = rng.randrange(total)
    fruit_indices = {(start + (i * total // fruit_count)) % total for i in range(fruit_count)}

    params.crown.radius = ranges.crown_radius.sample(rng)
    params.crown.half_height = ranges.crown_half_height.sample(rng)
    fruit_radius = params.berry_profile.r_base
    leaf_radius = params.leaf_profile.r_base
    params.berry_profile = exp_profile(
        k_base=ranges.fruit_stiffness.sample(rng),
        length=ranges.fruit_petiole_length.sample(rng),
        r_base=fruit_radius * ranges.petiole_radius_scale.sample(rng),
        taper=ranges.petiole_taper.sample(rng),
        damping_ratio=ranges.damping_ratio.sample(rng),
    )
    params.leaf_profile = logistic_profile(
        k_base=ranges.leaf_stiffness.sample(rng),
        length=ranges.leaf_petiole_length.sample(rng),
        r_base=leaf_radius * ranges.petiole_radius_scale.sample(rng),
        taper=ranges.petiole_taper.sample(rng),
        damping_ratio=ranges.damping_ratio.sample(rng),
    )
    params.leaf_blade.length = ranges.leaf_length.sample(rng)
    params.leaf_blade.width = ranges.leaf_width.sample(rng)
    params.leaf_blade.spread_angle = ranges.leaflet_spread.sample(rng)
    params.berry.r_base = ranges.berry_radius.sample(rng)

    params.leaves = []
    for i, azimuth in enumerate(azimuths):
        if i not in fruit_indices:
            params.leaves.append(LeafEntry(azimuth, ranges.leaf_elevation.sample(rng)))
            continue
        branch_count = rng.randint(*ranges.branches_per_truss)
        fractions = sorted(ranges.branch_fraction.sample(rng) for _ in range(branch_count))
        branches = []
        for fraction in fractions:
            has_sub = rng.random() < ranges.sub_branch_probability
            branches.append(BranchSpec(
                frac=fraction,
                ped_len=ranges.pedicel_length.sample(rng),
                droop_deg=ranges.pedicel_droop.sample(rng),
                berry_r_scale=ranges.berry_scale.sample(rng),
                has_sub=has_sub,
                sub_len=ranges.pedicel_length.sample(rng) * 0.7 if has_sub else 0.0,
                sub_droop_deg=ranges.pedicel_droop.sample(rng) if has_sub else 0.0,
                sub_berry_r_scale=ranges.berry_scale.sample(rng) if has_sub else 1.0,
            ))
        truss = TrussConfig(
            stem_len=ranges.stem_length.sample(rng),
            stem_r_base=ranges.stem_radius.sample(rng),
            stem_k_base=ranges.stem_stiffness.sample(rng),
            plane_az_deg=rng.uniform(0, 360),
            branches=branches,
            tip_berry_r_scale=ranges.berry_scale.sample(rng),
            tip_ped_len=ranges.pedicel_length.sample(rng) * 0.7,
        )
        params.leaves.append(LeafEntry(azimuth, ranges.fruit_elevation.sample(rng), truss))
    return params


def sample_plants(count: int, seed: int | None = None,
                  ranges: PlantRanges = PlantRanges()) -> list[PlantParams]:
    if count < 0:
        raise ValueError("count must be nonnegative")
    rng = random.Random(seed)
    return [sample_plant(rng, ranges) for _ in range(count)]

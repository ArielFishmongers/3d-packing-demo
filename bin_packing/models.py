from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple


@dataclass(frozen=True)
class Orientation:
    """One axis-aligned rotation of a cuboid — extents along x, y, z."""

    l: float  # extent along x
    w: float  # extent along y
    h: float  # extent along z

    @property
    def dims(self) -> tuple[float, float, float]:
        return (self.l, self.w, self.h)

    @property
    def volume(self) -> float:
        return self.l * self.w * self.h

    @property
    def base_area(self) -> float:
        return self.l * self.w


@dataclass
class Cuboid:
    """One item to be packed."""

    item_id: str
    dims: tuple[float, float, float]  # canonical (l, w, h); not tied to orientation
    weight: float = 1.0
    rotations_allowed: bool = True

    # Populated by preprocess_cuboids()
    orientations: list[Orientation] = field(default_factory=list)

    @property
    def volume(self) -> float:
        return self.dims[0] * self.dims[1] * self.dims[2]


@dataclass
class Placement:
    """A placed item's full description inside the container."""

    item_id: str
    orientation: Orientation
    position: tuple[float, float, float]  # (x, y, z) = bottom-front-left corner

    @property
    def x_max(self) -> float:
        return self.position[0] + self.orientation.l

    @property
    def y_max(self) -> float:
        return self.position[1] + self.orientation.w

    @property
    def z_max(self) -> float:
        return self.position[2] + self.orientation.h


@dataclass
class PackingResult:
    """Final result of one packing run."""

    placements: list[Placement]
    unpacked_ids: list[str]
    container_dims: tuple[float, float, float]

    @property
    def utilisation(self) -> float:
        packed_vol = sum(p.orientation.volume for p in self.placements)
        container_vol = (
            self.container_dims[0] * self.container_dims[1] * self.container_dims[2]
        )
        return packed_vol / container_vol if container_vol > 0 else 0.0

    @property
    def h_max(self) -> float:
        return max((p.z_max for p in self.placements), default=0.0)

    @property
    def support_quality(self) -> float:
        """Fraction of placed items out of all items (simple proxy)."""
        total = len(self.placements) + len(self.unpacked_ids)
        return len(self.placements) / total if total > 0 else 0.0


class Gene(NamedTuple):
    """One gene in a GA chromosome: an item and its preferred orientation index."""

    item_id: str
    orientation_idx: int


# A chromosome is an ordered list of genes
Chromosome = list[Gene]

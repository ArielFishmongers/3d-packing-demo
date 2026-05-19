from __future__ import annotations

import copy
from dataclasses import dataclass

from .models import Chromosome, Cuboid, PackingResult
from .heuristic import pack_with_heuristic


@dataclass
class FitnessWeights:
    """Weights for the fitness function components."""

    alpha: float = 1.0    # penalty for unused volume fraction (in [0, 1])
    beta: float = 1.0     # penalty for max height fraction (in [0, 1])
    gamma: float = 100.0  # reward per item placed (dominates across item counts)
    delta: float = 0.5    # reward for support quality


def evaluate_chromosome(
    chromosome: Chromosome,
    cuboid_map: dict[str, Cuboid],
    container_dims: tuple[int, int, int],
    weights: FitnessWeights | None = None,
) -> tuple[float, PackingResult]:
    """
    Decode a chromosome into an ordered list of cuboids, run the heuristic,
    and return (fitness_score, PackingResult).

    Fitness = gamma * item_count - alpha * unused_vol_fraction
              - beta * max_height_fraction + delta * support_quality

    Higher fitness is better. gamma is set large enough (default 100) so that
    each additional item placed dominates the alpha/beta penalty terms.

    The chromosome's orientation_idx biases the heuristic toward that rotation
    by placing the preferred orientation first in each cuboid's orientation list.
    """
    if weights is None:
        weights = FitnessWeights()

    ordered_cuboids: list[Cuboid] = []
    for gene in chromosome:
        c = cuboid_map[gene.item_id]
        idx = gene.orientation_idx % len(c.orientations)
        # Reorder: preferred orientation first, rest follow
        preferred = c.orientations[idx]
        reordered = [preferred] + [o for i, o in enumerate(c.orientations) if i != idx]
        c_copy = copy.copy(c)
        c_copy.orientations = reordered
        ordered_cuboids.append(c_copy)

    result = pack_with_heuristic(ordered_cuboids, container_dims, fail_fast=False)

    H = container_dims[2]
    item_count = len(result.placements)
    unused_vol_fraction = 1.0 - result.utilisation       # in [0, 1]
    max_height_fraction = result.h_max / H if H > 0 else 0.0
    support = result.support_quality

    fitness = (
        weights.gamma * item_count
        - weights.alpha * unused_vol_fraction
        - weights.beta * max_height_fraction
        + weights.delta * support
    )
    return fitness, result

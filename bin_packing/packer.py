from __future__ import annotations

from dataclasses import dataclass, field

from .genetic import GAConfig, GeneticOptimiser
from .heuristic import pack_with_heuristic
from .models import Cuboid, PackingResult
from .orientations import preprocess_cuboids
from .sequential import pack_sequential as _pack_sequential


@dataclass
class PackerConfig:
    use_ga: bool = True
    ga_config: GAConfig = field(default_factory=GAConfig)
    sort_by: str = "volume"    # "volume" | "longest_edge"
    precision: int = 1         # voxel scale factor for float dimensions


class Packer:
    """
    Public API for 3D bin packing.

    Usage::

        from bin_packing import Packer, Cuboid

        packer = Packer(container_dims=(100, 80, 60))
        cuboids = [Cuboid("box1", (10, 20, 15)), Cuboid("box2", (5, 5, 5))]
        result = packer.pack(cuboids)
        print(f"Utilisation: {result.utilisation:.1%}")
    """

    def __init__(
        self,
        container_dims: tuple[float, float, float],
        config: PackerConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or PackerConfig()
        self.seed = seed
        prec = self.config.precision
        self.container_dims_voxel: tuple[int, int, int] = (
            int(container_dims[0] * prec),
            int(container_dims[1] * prec),
            int(container_dims[2] * prec),
        )

    def _scale_cuboids(self, cuboids: list[Cuboid]) -> list[Cuboid]:
        """Scale float cuboid dimensions to integer voxel units."""
        prec = self.config.precision
        return [
            Cuboid(
                item_id=c.item_id,
                dims=tuple(d * prec for d in c.dims),  # type: ignore[arg-type]
                weight=c.weight,
                rotations_allowed=c.rotations_allowed,
            )
            for c in cuboids
        ]

    def pack(self, cuboids: list[Cuboid]) -> PackingResult:
        """
        Pack all cuboids into the container.

        Runs pre-processing (orientation generation + sort), then the
        extreme-point heuristic, and optionally the genetic algorithm.
        Returns the result with the highest utilisation.
        """
        # Scale float dimensions to integer voxels
        scaled = self._scale_cuboids(cuboids)

        # Pre-processing: orientation generation + sort
        sorted_cuboids = preprocess_cuboids(scaled, sort_by=self.config.sort_by)

        # Phase 1: heuristic baseline
        heuristic_result = pack_with_heuristic(sorted_cuboids, self.container_dims_voxel)

        if not self.config.use_ga:
            return heuristic_result

        # Phase 2: GA refinement
        optimiser = GeneticOptimiser(
            cuboids=sorted_cuboids,
            container_dims=self.container_dims_voxel,
            config=self.config.ga_config,
            seed=self.seed,
        )
        ga_result, _ = optimiser.run()

        if ga_result is None or ga_result.utilisation < heuristic_result.utilisation:
            return heuristic_result
        return ga_result

    def pack_sequential(
        self,
        cuboids: list[Cuboid],
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> PackingResult:
        """
        Pack cuboids in strict arrival order without sorting or skipping.

        Unlike :meth:`pack`, no pre-sorting is applied and no item is
        skipped. On the first item that cannot be placed the algorithm
        stops immediately; that item and all subsequent items are recorded
        in ``unpacked_ids``.

        Args:
            cuboids: Items in the order they must be packed.
            alpha:   Weight for bounding-box void volume in placement cost.
            beta:    Weight for maximum height in placement cost.

        Returns:
            PackingResult with the placed prefix and any unpacked suffix.
        """
        scaled = self._scale_cuboids(cuboids)
        return _pack_sequential(scaled, self.container_dims_voxel, alpha=alpha, beta=beta)

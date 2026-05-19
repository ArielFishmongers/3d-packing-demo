from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .models import Cuboid, PackingResult
from .packer import Packer, PackerConfig

PACKING_MODES = ("sequential", "heuristic", "ga")
PackingStrategy = Callable[
    [list[Cuboid], tuple[int, int, int], "AlgorithmConfig"],
    PackingResult,
]


@dataclass(frozen=True)
class AlgorithmConfig:
    mode: str = "sequential"
    precision: int = 1
    sort_by: str = "volume"
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in PACKING_MODES:
            raise ValueError(f"Unsupported algorithm mode: {self.mode}")
        if self.precision < 1:
            raise ValueError("precision must be at least 1")
        if self.sort_by not in {"volume", "longest_edge"}:
            raise ValueError(f"Unsupported sort key: {self.sort_by}")


def _build_packer(
    container_dims: tuple[int, int, int],
    algorithm: AlgorithmConfig,
    *,
    use_ga: bool,
) -> Packer:
    config = PackerConfig(
        use_ga=use_ga,
        sort_by=algorithm.sort_by,
        precision=algorithm.precision,
    )
    return Packer(container_dims, config=config, seed=algorithm.seed)


def _pack_sequential(
    cuboids: list[Cuboid],
    container_dims: tuple[int, int, int],
    algorithm: AlgorithmConfig,
) -> PackingResult:
    packer = _build_packer(container_dims, algorithm, use_ga=False)
    return packer.pack_sequential(cuboids)


def _pack_heuristic(
    cuboids: list[Cuboid],
    container_dims: tuple[int, int, int],
    algorithm: AlgorithmConfig,
) -> PackingResult:
    packer = _build_packer(container_dims, algorithm, use_ga=False)
    return packer.pack(cuboids)


def _pack_ga(
    cuboids: list[Cuboid],
    container_dims: tuple[int, int, int],
    algorithm: AlgorithmConfig,
) -> PackingResult:
    packer = _build_packer(container_dims, algorithm, use_ga=True)
    return packer.pack(cuboids)


ALGORITHM_REGISTRY: dict[str, PackingStrategy] = {
    "sequential": _pack_sequential,
    "heuristic": _pack_heuristic,
    "ga": _pack_ga,
}


def pack_once(
    cuboids: list[Cuboid],
    container_dims: tuple[int, int, int],
    algorithm: AlgorithmConfig | None = None,
) -> PackingResult:
    selected = algorithm or AlgorithmConfig()
    return ALGORITHM_REGISTRY[selected.mode](cuboids, container_dims, selected)


def pack_multi_container(
    cuboids: list[Cuboid],
    container_dims: tuple[int, int, int],
    *,
    algorithm: AlgorithmConfig | None = None,
    quiet: bool = False,
) -> tuple[list[PackingResult], list[str]]:
    """Pack cuboids into as many containers as needed."""
    selected = algorithm or AlgorithmConfig()
    results: list[PackingResult] = []
    remaining = list(cuboids)
    permanently_skipped: list[str] = []
    warnings: list[str] = []

    def warn(message: str) -> None:
        if quiet:
            warnings.append(message)
        else:
            print(message)

    while remaining:
        result = pack_once(remaining, container_dims, selected)

        if not result.placements:
            stuck = remaining[0]
            permanently_skipped.append(stuck.item_id)
            warn(
                f"  Warning: item {stuck.item_id} (dims {stuck.dims}) cannot fit in "
                f"{container_dims[0]}×{container_dims[1]}×{container_dims[2]} cm — skipping"
            )
            remaining = remaining[1:]
            continue

        results.append(result)
        if not result.unpacked_ids:
            break

        unpacked = set(result.unpacked_ids)
        remaining = [cuboid for cuboid in remaining if cuboid.item_id in unpacked]

    if permanently_skipped:
        warn(
            f"  Warning: {len(permanently_skipped)} item(s) permanently skipped "
            f"(too large for container)"
        )

    return results, warnings

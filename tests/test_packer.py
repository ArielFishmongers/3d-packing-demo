import time
import random

import pytest

from bin_packing import Cuboid, Packer, PackerConfig
from bin_packing.genetic import GAConfig
from bin_packing.occupancy import OccupancyGrid


def make_random_cuboids(n, seed=0, max_dim=5):
    rng = random.Random(seed)
    return [
        Cuboid(f"box{i}", (rng.randint(1, max_dim), rng.randint(1, max_dim), rng.randint(1, max_dim)))
        for i in range(n)
    ]


def assert_no_overlap(result, container_dims):
    L, W, H = (int(d) for d in container_dims)
    grid = OccupancyGrid(L, W, H)
    for p in result.placements:
        x, y, z = (int(c) for c in p.position)
        l, w, h = int(p.orientation.l), int(p.orientation.w), int(p.orientation.h)
        assert grid.is_free(x, y, z, l, w, h), f"Overlap detected for {p.item_id}"
        grid.place(p.item_id, x, y, z, l, w, h)


# ------------------------------------------------------------------
# Heuristic-only tests
# ------------------------------------------------------------------

def test_heuristic_only_basic():
    packer = Packer((10, 10, 10), PackerConfig(use_ga=False))
    result = packer.pack([Cuboid("a", (3, 3, 3))])
    assert len(result.placements) == 1
    assert result.utilisation > 0


def test_heuristic_only_no_overlap():
    cuboids = make_random_cuboids(10, seed=1)
    packer = Packer((10, 10, 10), PackerConfig(use_ga=False))
    result = packer.pack(cuboids)
    assert_no_overlap(result, (10, 10, 10))


def test_heuristic_only_speed_n50():
    cuboids = make_random_cuboids(50, seed=2)
    packer = Packer((20, 20, 20), PackerConfig(use_ga=False))
    start = time.time()
    result = packer.pack(cuboids)
    elapsed = time.time() - start
    assert elapsed < 2.0, f"Heuristic for n=50 took {elapsed:.2f}s (limit 2s)"
    assert result.utilisation > 0


def test_heuristic_only_identical_cubes():
    cuboids = [Cuboid(f"c{i}", (2, 2, 2)) for i in range(8)]
    packer = Packer((4, 4, 4), PackerConfig(use_ga=False))
    result = packer.pack(cuboids)
    assert len(result.placements) == 8
    assert result.utilisation == pytest.approx(1.0)


# ------------------------------------------------------------------
# GA tests
# ------------------------------------------------------------------

def test_packer_with_ga_no_overlap():
    cuboids = make_random_cuboids(8, seed=3)
    config = PackerConfig(
        use_ga=True,
        ga_config=GAConfig(population_size=10, n_generations=5, plateau_patience=3),
    )
    packer = Packer((10, 10, 10), config, seed=42)
    result = packer.pack(cuboids)
    assert_no_overlap(result, (10, 10, 10))


def test_ga_utilisation_at_least_as_good_as_heuristic():
    cuboids = make_random_cuboids(10, seed=4)
    heuristic_config = PackerConfig(use_ga=False)
    ga_config = PackerConfig(
        use_ga=True,
        ga_config=GAConfig(population_size=15, n_generations=10, plateau_patience=5),
    )
    h_result = Packer((10, 10, 10), heuristic_config).pack(cuboids)
    ga_result = Packer((10, 10, 10), ga_config, seed=0).pack(cuboids)
    assert ga_result.utilisation >= h_result.utilisation - 1e-9


def test_utilisation_above_threshold_20_items():
    cuboids = make_random_cuboids(20, seed=5, max_dim=4)
    config = PackerConfig(
        use_ga=True,
        ga_config=GAConfig(population_size=20, n_generations=20, plateau_patience=10),
    )
    packer = Packer((10, 10, 10), config, seed=1)
    result = packer.pack(cuboids)
    assert result.utilisation > 0.20


# ------------------------------------------------------------------
# Float dimensions with precision
# ------------------------------------------------------------------

def test_float_dimensions_precision():
    # 8 boxes of (1.0, 1.0, 1.0) in a (2.0, 2.0, 2.0) container at precision=2
    # Each box → 2×2×2 voxels = 8 voxels; container → 4×4×4 = 64 voxels
    # 8 boxes × 8 voxels = 64 → 100 % utilisation
    cuboids = [Cuboid(f"h{i}", (1.0, 1.0, 1.0)) for i in range(8)]
    config = PackerConfig(use_ga=False, precision=2)
    packer = Packer((2.0, 2.0, 2.0), config)
    result = packer.pack(cuboids)
    assert len(result.placements) == 8
    assert result.utilisation == pytest.approx(1.0)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_empty_input():
    packer = Packer((10, 10, 10), PackerConfig(use_ga=False))
    result = packer.pack([])
    assert result.placements == []
    assert result.unpacked_ids == []


def test_box_larger_than_container_skipped():
    cuboids = [Cuboid("giant", (20, 20, 20))]
    packer = Packer((10, 10, 10), PackerConfig(use_ga=False))
    result = packer.pack(cuboids)
    assert len(result.placements) == 0
    assert "giant" in result.unpacked_ids


def test_packer_large_n_completes(benchmark=None):
    cuboids = make_random_cuboids(100, seed=6, max_dim=3)
    config = PackerConfig(use_ga=False)
    packer = Packer((20, 20, 20), config)
    start = time.time()
    result = packer.pack(cuboids)
    elapsed = time.time() - start
    assert elapsed < 30.0, f"n=100 heuristic took {elapsed:.2f}s"
    assert result.utilisation > 0

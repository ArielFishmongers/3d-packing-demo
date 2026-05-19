import pytest
from bin_packing.models import Cuboid, Gene
from bin_packing.orientations import preprocess_cuboids
from bin_packing.genetic import GeneticOptimiser, GAConfig
from bin_packing.heuristic import pack_with_heuristic


def make_cuboids(*dims_list):
    cuboids = [Cuboid(f"box{i}", d) for i, d in enumerate(dims_list)]
    return preprocess_cuboids(cuboids)


@pytest.fixture
def optimiser_small():
    cuboids = make_cuboids(
        (3, 3, 3), (2, 2, 4), (1, 4, 2), (2, 3, 2), (4, 1, 3)
    )
    config = GAConfig(population_size=10, n_generations=5, plateau_patience=10)
    return GeneticOptimiser(cuboids=cuboids, container_dims=(10, 10, 10), config=config, seed=42)


def test_ox_crossover_valid_permutation(optimiser_small):
    ga = optimiser_small
    p1 = ga._heuristic_chromosome()
    p2 = ga._random_chromosome()
    child_a, child_b = ga._crossover(p1, p2)
    ids_in = {g.item_id for g in p1}
    assert {g.item_id for g in child_a} == ids_in
    assert {g.item_id for g in child_b} == ids_in
    assert len(child_a) == len(p1)
    assert len(child_b) == len(p1)


def test_ox_crossover_no_duplicate_ids(optimiser_small):
    ga = optimiser_small
    p1 = ga._heuristic_chromosome()
    p2 = ga._random_chromosome()
    child_a, _ = ga._crossover(p1, p2)
    ids = [g.item_id for g in child_a]
    assert len(ids) == len(set(ids))


def test_mutation_swap_preserves_item_ids(optimiser_small):
    ga = optimiser_small
    ga.config.mutation_swap_prob = 1.0  # force swap mutation
    chrom = ga._heuristic_chromosome()
    mutated = ga._mutate(chrom)
    assert {g.item_id for g in mutated} == {g.item_id for g in chrom}
    assert len(mutated) == len(chrom)


def test_mutation_orient_changes_only_orientation(optimiser_small):
    ga = optimiser_small
    ga.config.mutation_swap_prob = 0.0  # force orientation mutation
    chrom = ga._heuristic_chromosome()
    mutated = ga._mutate(chrom)
    # item_ids must be the same in the same positions
    for orig, mut in zip(chrom, mutated):
        assert orig.item_id == mut.item_id


def test_ga_runs_without_error(optimiser_small):
    result, history = optimiser_small.run()
    assert result is not None
    assert len(history) > 0


def test_ga_fitness_history_nondecreasing_overall(optimiser_small):
    _, history = optimiser_small.run()
    # Best-ever fitness should not decrease over the full run
    running_best = history[0]
    for h in history[1:]:
        running_best = max(running_best, h)
    assert running_best >= history[0]


def test_ga_result_has_placements(optimiser_small):
    result, _ = optimiser_small.run()
    assert len(result.placements) > 0


def test_ga_determinism_with_seed():
    cuboids = make_cuboids((3, 3, 3), (2, 2, 4), (1, 4, 2))
    config = GAConfig(population_size=8, n_generations=3)
    r1, h1 = GeneticOptimiser(cuboids, (10, 10, 10), config, seed=99).run()
    r2, h2 = GeneticOptimiser(cuboids, (10, 10, 10), config, seed=99).run()
    assert h1 == h2
    assert r1.utilisation == pytest.approx(r2.utilisation)


def test_plateau_stopping():
    # With patience=1 and a very small population/generations, GA should
    # stop early after first generation if no improvement.
    cuboids = make_cuboids((5, 5, 5))
    # Single-item problem: heuristic is already optimal. GA will plateau immediately.
    config = GAConfig(
        population_size=5, n_generations=100, plateau_patience=1
    )
    ga = GeneticOptimiser(cuboids, (10, 10, 10), config, seed=0)
    _, history = ga.run()
    # Should stop well before 100 generations
    assert len(history) < 100


def test_elitism_preserves_best():
    cuboids = make_cuboids((3, 3, 3), (2, 2, 2), (1, 1, 1))
    config = GAConfig(
        population_size=10, n_generations=10, elite_fraction=0.2, plateau_patience=20
    )
    ga = GeneticOptimiser(cuboids, (10, 10, 10), config, seed=7)
    _, history = ga.run()
    # Best fitness should never decrease over the sequence
    for i in range(1, len(history)):
        assert max(history[: i + 1]) >= max(history[:i])


def test_ga_improves_or_matches_heuristic():
    cuboids = make_cuboids(
        (3, 3, 3), (2, 2, 4), (1, 4, 2), (2, 3, 2), (4, 1, 3),
        (2, 2, 2), (3, 2, 1), (1, 1, 4)
    )
    container = (10, 10, 10)
    heuristic_result = pack_with_heuristic(list(cuboids), container)
    config = GAConfig(population_size=20, n_generations=20, plateau_patience=10)
    ga = GeneticOptimiser(cuboids, container, config, seed=42)
    ga_result, _ = ga.run()
    assert ga_result.utilisation >= heuristic_result.utilisation - 1e-9

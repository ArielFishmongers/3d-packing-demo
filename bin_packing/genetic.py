from __future__ import annotations

import random
from dataclasses import dataclass, field

from .fitness import FitnessWeights, evaluate_chromosome
from .models import Chromosome, Cuboid, Gene, PackingResult


@dataclass
class GAConfig:
    """Hyperparameters for the genetic algorithm."""

    population_size: int = 50
    n_generations: int = 200
    elite_fraction: float = 0.1       # top fraction survive unchanged
    tournament_size: int = 3
    crossover_prob: float = 0.85
    mutation_prob: float = 0.15       # per chromosome
    mutation_swap_prob: float = 0.5   # within mutation: swap vs orientation change
    plateau_patience: int = 30        # stop if no improvement for N generations
    weights: FitnessWeights = field(default_factory=FitnessWeights)


class GeneticOptimiser:
    """
    Genetic algorithm that refines the packing order and orientations to
    maximise the fitness function.
    """

    def __init__(
        self,
        cuboids: list[Cuboid],
        container_dims: tuple[int, int, int],
        config: GAConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.cuboids = cuboids
        self.cuboid_map: dict[str, Cuboid] = {c.item_id: c for c in cuboids}
        self.container_dims = container_dims
        self.config = config or GAConfig()
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Chromosome construction
    # ------------------------------------------------------------------

    def _random_chromosome(self) -> Chromosome:
        items = list(self.cuboids)
        self.rng.shuffle(items)
        return [
            Gene(
                item_id=c.item_id,
                orientation_idx=self.rng.randrange(len(c.orientations)),
            )
            for c in items
        ]

    def _heuristic_chromosome(self) -> Chromosome:
        """Chromosome reflecting the sorted heuristic order (orientation 0)."""
        return [Gene(item_id=c.item_id, orientation_idx=0) for c in self.cuboids]

    def _initial_population(self) -> list[Chromosome]:
        pop: list[Chromosome] = [self._heuristic_chromosome()]
        while len(pop) < self.config.population_size:
            pop.append(self._random_chromosome())
        return pop

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _tournament_select(
        self,
        population: list[Chromosome],
        scores: list[float],
    ) -> Chromosome:
        k = min(self.config.tournament_size, len(population))
        indices = self.rng.sample(range(len(population)), k)
        best_idx = max(indices, key=lambda i: scores[i])
        return population[best_idx]

    # ------------------------------------------------------------------
    # Crossover — Order Crossover (OX)
    # ------------------------------------------------------------------

    def _crossover(
        self, parent_a: Chromosome, parent_b: Chromosome
    ) -> tuple[Chromosome, Chromosome]:
        n = len(parent_a)
        if n < 2:
            return list(parent_a), list(parent_b)
        cut1, cut2 = sorted(self.rng.sample(range(n), 2))
        child_a = self._ox_child(parent_a, parent_b, cut1, cut2)
        child_b = self._ox_child(parent_b, parent_a, cut1, cut2)
        return child_a, child_b

    def _ox_child(
        self,
        p1: Chromosome,
        p2: Chromosome,
        cut1: int,
        cut2: int,
    ) -> Chromosome:
        n = len(p1)
        child: list[Gene | None] = [None] * n
        # Copy the segment [cut1, cut2] from p1
        segment_ids: set[str] = set()
        for i in range(cut1, cut2 + 1):
            child[i] = p1[i]
            segment_ids.add(p1[i].item_id)
        # Fill remaining positions in order from p2, skipping duplicates
        p2_remaining = [g for g in p2 if g.item_id not in segment_ids]
        fill_positions = list(range(cut2 + 1, n)) + list(range(0, cut1))
        for pos, gene in zip(fill_positions, p2_remaining):
            child[pos] = gene
        return child  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        chrom = list(chromosome)
        if self.rng.random() < self.config.mutation_swap_prob:
            # Swap two randomly chosen items
            if len(chrom) >= 2:
                i, j = self.rng.sample(range(len(chrom)), 2)
                chrom[i], chrom[j] = chrom[j], chrom[i]
        else:
            # Change one item's orientation index
            i = self.rng.randrange(len(chrom))
            c = self.cuboid_map[chrom[i].item_id]
            new_orient = self.rng.randrange(len(c.orientations))
            chrom[i] = Gene(item_id=chrom[i].item_id, orientation_idx=new_orient)
        return chrom

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> tuple[PackingResult, list[float]]:
        """
        Run the GA and return (best_PackingResult, fitness_history).

        fitness_history[g] is the best fitness achieved in generation g.
        """
        cfg = self.config
        population = self._initial_population()
        best_result: PackingResult | None = None
        best_fitness: float = float("-inf")
        fitness_history: list[float] = []
        no_improve_count = 0

        for _ in range(cfg.n_generations):
            # Evaluate all chromosomes
            scored: list[tuple[float, Chromosome, PackingResult]] = []
            for chrom in population:
                fit, result = evaluate_chromosome(
                    chrom, self.cuboid_map, self.container_dims, cfg.weights
                )
                scored.append((fit, chrom, result))
            scored.sort(key=lambda t: t[0], reverse=True)

            gen_best_fit = scored[0][0]
            fitness_history.append(gen_best_fit)

            if gen_best_fit > best_fitness:
                best_fitness = gen_best_fit
                best_result = scored[0][2]
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= cfg.plateau_patience:
                break

            # Elitism: keep top N unchanged
            n_elite = max(1, int(cfg.elite_fraction * len(population)))
            new_population: list[Chromosome] = [t[1] for t in scored[:n_elite]]

            scores = [t[0] for t in scored]
            chroms = [t[1] for t in scored]

            # Breed until population is refilled
            while len(new_population) < cfg.population_size:
                pa = self._tournament_select(chroms, scores)
                pb = self._tournament_select(chroms, scores)
                if self.rng.random() < cfg.crossover_prob:
                    ca, cb = self._crossover(pa, pb)
                else:
                    ca, cb = list(pa), list(pb)
                if self.rng.random() < cfg.mutation_prob:
                    ca = self._mutate(ca)
                if self.rng.random() < cfg.mutation_prob:
                    cb = self._mutate(cb)
                new_population.extend([ca, cb])

            population = new_population[: cfg.population_size]

        return best_result, fitness_history

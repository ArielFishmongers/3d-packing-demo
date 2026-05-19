# 3D Bin Packing

Hybrid extreme-point heuristic with genetic search for packing 3D cuboids into a container.

## Features

- Three packing modes: **heuristic** (greedy extreme-point), **hybrid GA** (heuristic + genetic refinement), and **sequential** (order-preserving).
- Up to six axis-aligned rotations per item; per-item rotation lock supported.
- Partial-support stability: at least 50% of a box's base must rest on existing support or the floor.
- Voxel-grid occupancy with configurable float-to-integer precision scaling.
- Interactive 3D visualisation (static and drop animation) via Plotly.
- Configurable GA (population, generations, plateau patience, weights for utilisation / height / support).

## Installation

Requires Python ≥ 3.11.

```bash
pip install -e .              # core (numpy only)
pip install -e ".[viz]"       # + Plotly / matplotlib for visualisation
pip install -e ".[dev]"       # + pytest, ruff, mypy, hypothesis
```

## Quick start

```python
from bin_packing import Cuboid, Packer, PackerConfig
from bin_packing.genetic import GAConfig

container = (10, 10, 5)
items = [
    Cuboid("pallet_A", dims=(8, 6, 5)),
    Cuboid("pallet_B", dims=(6, 6, 4)),
    Cuboid("box_1",    dims=(4, 4, 4)),
    Cuboid("flat_1",   dims=(6, 5, 1)),
]

packer = Packer(container, PackerConfig(use_ga=True, ga_config=GAConfig()), seed=42)
result = packer.pack(items)

print(f"Packed:      {len(result.placements)}/{len(items)}")
print(f"Utilisation: {result.utilisation:.1%}")
print(f"Max height:  {result.h_max}")
print(f"Unplaced:    {result.unpacked_ids}")
```

`Packer.pack()` returns a [PackingResult](bin_packing/models.py#L67) with `placements`, `unpacked_ids`, `container_dims`, and computed properties `utilisation`, `h_max`, and `support_quality`. A runnable end-to-end demo lives in [examples/basic_usage.py](examples/basic_usage.py).

## Packing modes

| Mode | Call | When to use |
|------|------|-------------|
| Heuristic only | `Packer(c, PackerConfig(use_ga=False)).pack(items)` | Fast baseline; items sorted by volume or longest edge. |
| Hybrid (default) | `Packer(c).pack(items)` | Best utilisation; runs the heuristic then refines with GA, returns whichever is better. |
| Sequential | `Packer(c).pack_sequential(items)` | Preserves arrival order, no reordering. Stops at the first item that does not fit; all later items go in `unpacked_ids`. |

Sequential mode is specified in [SEQUENTIAL_PACKING_PLAN.md](SEQUENTIAL_PACKING_PLAN.md).

## Visualisation

```python
from bin_packing.visualisation import plot_packing, animate_packing

plot_packing(result).show()       # static interactive 3D view
animate_packing(result).show()    # boxes drop in packing order
```

Both return a `plotly.graph_objects.Figure`; use `.write_html("out.html")` to save. Requires the `viz` extra.

## How it works

1. **Pre-processing.** Generate up to six unique orientations per item, then sort by volume (or longest edge).
2. **Extreme-point heuristic.** Maintain a list of candidate placement corners — extreme points — starting at `(0, 0, 0)`. For each item, try every `(point, orientation)` pair, reject those that overlap or fail the partial-support check, and score the rest by residual bounding-box volume (Best-Fit). Place at the best score, then generate new extreme points at the item's three forward faces and prune dominated points.
3. **Genetic refinement.** Encode a solution as a sequence of `(item_id, orientation_idx)` genes. The GA evolves the population with tournament selection, crossover, and low-rate mutation, scoring each chromosome by `α·utilisation − β·h_max + γ·support`. The hybrid call returns the better of heuristic and GA.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full literature review, problem framing, and detailed algorithm rationale.

## Project structure

```
bin_packing/
  packer.py            Packer / PackerConfig — public entry point
  models.py            Cuboid, Orientation, Placement, PackingResult, Gene, Chromosome
  heuristic.py         pack_with_heuristic — extreme-point greedy placement
  extreme_points.py    EP generation and Key-Point pruning
  genetic.py           GeneticOptimiser, GAConfig
  fitness.py           Chromosome fitness evaluation
  sequential.py        pack_sequential — order-preserving mode
  occupancy.py         OccupancyGrid (overlap and partial-support checks)
  orientations.py      Rotation enumeration and pre-processing sort
  visualisation.py     plot_packing, animate_packing (Plotly)
  exceptions.py        PackingFailure, InfeasibleItem
examples/              Runnable demos (start with basic_usage.py)
scripts/               Benchmarks and utility scripts
tests/                 Pytest suite
```

## Development

```bash
pytest            # run the test suite
ruff check .      # lint
mypy bin_packing  # type-check
```

## Roadmap

- Reinforcement-learning refinement over Key-Point placements (the `rl` extra wires Torch/Gymnasium; integration pending — see [PROJECT_PLAN.md](PROJECT_PLAN.md)).

## References

- [PROJECT_PLAN.md](PROJECT_PLAN.md) — full problem definition, literature review, and algorithm design.
- [SEQUENTIAL_PACKING_PLAN.md](SEQUENTIAL_PACKING_PLAN.md) — sequential / order-preserving mode specification.

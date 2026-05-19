# Sequential Cuboid Packing Plan

## Scope and repository target

This document is the standalone implementation spec for a **sequential, order-preserving** packing mode.

- It is **not** a replacement for the current hybrid packer behavior.
- It should be implemented as a **separate path** alongside the current `Packer.pack()` flow.
- The current hybrid heuristic and GA behavior should remain unchanged.
- Reinforcement learning and GA-based sequential tuning are future extensions and are **out of scope for the initial implementation** described here.

Repository integration target:

- Reuse orientation generation from [bin_packing/orientations.py](/Users/ariel/Documents/Personal Projects/3D-Packing/bin_packing/orientations.py).
- Reuse occupancy checks and support checks from [bin_packing/occupancy.py](/Users/ariel/Documents/Personal Projects/3D-Packing/bin_packing/occupancy.py) and [bin_packing/heuristic.py](/Users/ariel/Documents/Personal Projects/3D-Packing/bin_packing/heuristic.py).
- Add a dedicated sequential implementation path rather than changing the semantics of the existing heuristic sorter in [bin_packing/packer.py](/Users/ariel/Documents/Personal Projects/3D-Packing/bin_packing/packer.py).
- Add focused tests for the sequential behavior without weakening the current heuristic or GA test suite.

## Problem definition

We consider a sequence of $n$ rectangular cuboids with given dimensions $(l_i, w_i, h_i)$ for $i = 1, \dots, n$. A fixed container of dimensions $(L, W, H)$ is available. The objective is to pack the cuboids into the container in **exactly the order provided**, with the following constraints:

- Each cuboid must be placed axis-aligned inside the container; up to six rotations are possible for each cuboid, although some cuboids may have a restricted subset of allowable orientations.
- Items must not overlap and must fit completely within the container boundaries.
- Once a cuboid is placed it cannot be moved or reordered.
- If a cuboid cannot be placed anywhere in the container when it arrives, the packing process **stops immediately** and the remaining items are not packed.
- The primary objective is to maximise the number of items packed under this stop-on-failure rule.
- Among placements that keep the sequence alive, the secondary objective is to minimise internal voids and the maximum height of the packed configuration.

Because three-dimensional bin packing is NP-hard, exact optimisation is computationally infeasible for large instances. Consequently, we rely on heuristic techniques to obtain high-quality solutions efficiently.

## Core decisions

1. **Order is fixed**
   - No pre-sorting is allowed.
   - No item reordering is allowed.
   - No skip flag is allowed in v1.

2. **State representation**
   - The **occupancy grid is the source of truth** for feasibility, support, and simulation.
   - A maintained free-space box list is **not required in v1**. This avoids multiple valid split implementations and keeps the spec deterministic.
   - Extreme points are still used as the candidate placement set.

3. **Support rule**
   - Support checking is enabled in v1.
   - A placement above the floor is valid only if the supported area fraction is at least `0.5`.
   - This must match the behavior already implemented in [bin_packing/heuristic.py](/Users/ariel/Documents/Personal Projects/3D-Packing/bin_packing/heuristic.py).

4. **Output semantics**
   - Return a `PackingResult`.
   - `placements` contains the placed prefix of the sequence in arrival order.
   - `unpacked_ids` contains the first unplaceable item and every remaining item after it, preserving original order.

## Candidate generation and scoring

### Extreme points

Maintain an extreme-point list $EP$.

- Initialise $EP = [(0,0,0)]$.
- After placing a cuboid at position $(x_i, y_i, z_i)$ with oriented dimensions $(l, w, h)$, generate:
  - $(x_i + l, y_i, z_i)$
  - $(x_i, y_i + w, z_i)$
  - $(x_i, y_i, z_i + h)$
- Discard any candidate that:
  - lies outside the container,
  - falls inside already occupied volume,
  - or is dominated by another extreme point using the existing dominance rule.

### Feasibility checks

For each cuboid $i$, for each extreme point $p = (x_p, y_p, z_p)$ in $EP$, and for each allowed orientation $o = (l, w, h)$:

1. Boundary check:
   - Require $x_p + l \leq L$, $y_p + w \leq W$, and $z_p + h \leq H$.
2. Overlap check:
   - Require the occupancy region to be entirely free.
3. Support check:
   - If $z_p = 0$, support is valid.
   - Otherwise require supported area fraction $\geq 0.5$.

Any candidate that fails one of these checks is discarded before scoring.

### Exact placement cost

The placement cost must be computed from a simulated occupancy-grid copy.

For each feasible candidate:

1. Copy the current occupancy grid.
2. Simulate the cuboid placement on the copy.
3. Compute:
   - $h_{max}$: the maximum occupied height after placement.
   - $x_{max}$: the maximum occupied extent in the $x$ direction after placement.
   - $y_{max}$: the maximum occupied extent in the $y$ direction after placement.
   - $V_{packed}$: total occupied volume after placement.
   - $V_{bbox} = x_{max} \cdot y_{max} \cdot h_{max}$.
   - $V_{void} = V_{bbox} - V_{packed}$.
4. Score the candidate with:

$$
c = \alpha \cdot V_{void} + \beta \cdot h_{max}
$$

where:

- $\alpha > 0$ and $\beta > 0$
- default v1 weights are $\alpha = 1.0$ and $\beta = 1.0$

This definition is deliberate:

- it avoids the ambiguous "sum of all remaining free volume in the container" metric, which is constant for a fixed item volume,
- and it makes compaction measurable and placement-dependent.

### Deterministic tie-break order

If multiple candidates have equal cost within exact arithmetic, break ties in this order:

1. Lower $h_{max}$
2. Lower $V_{void}$
3. Higher support fraction
4. Lower placement coordinate in lexicographic order $(z_p, y_p, x_p)$
5. Lower orientation index in the cuboid's generated orientation list

This tie-break order is mandatory so the implementation is deterministic.

## Sequential packing algorithm

1. Initialisation
   - Generate allowed orientations for each cuboid in sequence order.
   - Initialise the occupancy grid for the container.
   - Initialise $EP = [(0,0,0)]$.
   - Initialise `placements = []`.
   - Initialise `unpacked_ids = []`.
   - Set $\alpha = 1.0$ and $\beta = 1.0$ unless explicitly overridden.

2. Process cuboids in the provided order
   - For the current cuboid, evaluate every feasible `(extreme point, orientation)` pair using the scoring rule above.
   - Select the best candidate using the mandatory tie-break rules.
   - If no candidate is feasible:
     - append the current cuboid's ID and all remaining cuboid IDs to `unpacked_ids`,
     - stop the algorithm immediately,
     - return the result.
   - Otherwise:
     - place the cuboid into the real occupancy grid,
     - append its placement to `placements`,
     - remove the used extreme point,
     - generate and prune new extreme points,
     - continue to the next cuboid.

3. Return result
   - Return a `PackingResult` containing:
     - the placed prefix,
     - the unpacked suffix if packing stopped early,
     - the container dimensions.

## Repository implementation notes

The intended v1 implementation should follow these repository boundaries:

1. Orientation generation
   - Reuse `unique_orientations()` and existing rotation restriction handling in [bin_packing/orientations.py](/Users/ariel/Documents/Personal Projects/3D-Packing/bin_packing/orientations.py).

2. Sequential heuristic
   - Implement a dedicated sequential function, for example `pack_sequential(...)`, rather than overloading the existing sorted heuristic semantics.
   - This function should live in a dedicated module or in [bin_packing/heuristic.py](/Users/ariel/Documents/Personal Projects/3D-Packing/bin_packing/heuristic.py) under a clearly separate entry point.

3. Public API
   - Do not silently change the meaning of `Packer.pack()`.
   - If the sequential mode is exposed publicly, it should be opt-in through a separate config flag or a separate public method.

4. Failure behavior
   - Current heuristic code can skip unpackable items when `fail_fast=False`.
   - The sequential implementation must **not** skip and continue.
   - It must stop on the first unplaceable cuboid and mark the remaining suffix as unpacked.

## Test expectations

Add or update tests so the sequential implementation proves the following:

1. **Order is preserved**
   - No sorting occurs before packing.

2. **Stop-on-failure behavior**
   - If item `k` cannot be placed, items `k+1...n` are not attempted.
   - `placements` contains only the prefix `0...k-1`.
   - `unpacked_ids` contains `k...n` in sequence order.

3. **Feasibility guarantees**
   - No overlap.
   - No out-of-bounds placements.
   - Support threshold is enforced for above-floor placements.

4. **Determinism**
   - Repeated runs on the same input produce the same placements.

5. **Tie-break coverage**
   - At least one test exercises equal-cost candidates and verifies the required tie-break order.

6. **Compatibility**
   - Existing tests for the current hybrid heuristic and GA continue to pass unchanged.

## Future extensions, not part of v1

1. Sequential GA
   - A GA may optimise orientation choices while preserving arrival order.
   - Skip flags are explicitly out of scope until the stop-on-failure baseline exists and is validated.

2. Reinforcement learning
   - A policy network may later choose placements from the extreme-point candidate set.
   - RL is not implemented today and is not required for the first sequential packer.

## Discussion

This sequential extreme-point heuristic is intended to be implementable without hidden design decisions. It trades some global optimality for deterministic behavior, clear failure semantics, and good practical performance under a strict arrival-order constraint. The occupancy grid is the authoritative state, the cost function is explicitly defined, tie-breaking is mandatory, and integration with the current repository is conservative: add the sequential mode without redefining the existing hybrid packer.

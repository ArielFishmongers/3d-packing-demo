# Optimising the packing of cuboids in a container

## Problem definition

We are given a set of $n$ cuboids (rectangular boxes) with fixed dimensions $(l_i, w_i, h_i)$ for $i \in \{1,\dots,n\}$. The task is to place all cuboids into a larger container with dimensions $(L, W, H)$ such that:

- Each item must be axis-aligned inside the container, but rotations of the box are allowed; there are six possible orientations.
- Items must not overlap and must lie entirely inside the container.
- The goal is to minimise unused space (equivalently, maximise the total packed volume) or minimise the height/number of containers used.

This problem is an instance of three-dimensional bin packing and is NP-hard. Research on approximation algorithms shows that even deciding whether all items fit into a unit cube is NP-hard; therefore there is a polynomial-time approximation hardness of $2$ for 3D bin packing. Recent theoretical work provides improved approximation ratios, for example $6$ for packing into a fixed-size bin and $3+\varepsilon$ for the minimum volume bounding box, but exact optimal solutions are impractical for large instances. Consequently, practitioners rely on heuristic and metaheuristic algorithms. A recent overview explains that exact methods guarantee optimality but have high computational costs, while heuristics quickly construct approximate solutions and metaheuristics iteratively refine them to balance solution quality and runtime.

In some applications the set of cuboids may exceed the capacity of a single container. The objective then is to pack as many cuboids as possible into a fixed container while minimising the unused volume of the resulting arrangement. This adds a knapsack-like aspect to the problem: the algorithm must choose a subset of items that yields the best combination of number of items and space utilisation.

To reduce empty space while satisfying real-world constraints (stability, partial support, weight stacking), our algorithm combines extreme-point heuristics with a genetic algorithm (GA) and optional reinforcement-learning (RL) refinements.

## Key techniques from the literature

1. Extreme point representation - An extreme point is a corner in the remaining free space where a box can be placed without overlapping others. The Extreme Point (EP) rule generates a list of candidate positions by projecting the faces of placed boxes and container boundaries. EP-based heuristics improve space utilisation compared with naive placement and can be used in modified First-Fit or Best-Fit algorithms.
2. Extreme-point priority sorting and partial support - A 2024 hybrid DRL algorithm prioritises extreme points by the residual volume of free space after placement. It also uses a 3-D grid state representation to capture occupied and empty cells and introduces partial support constraints: only part of the base of a box must be supported as long as more than half of its projected area lies over existing support or the container floor. This improves stability while permitting overhangs.
3. Key points to reduce the action space - In DRL settings, the action space (possible placements) can be large. Placing boxes only at so-called Key Points (a reduced set of extreme points) dramatically reduces complexity and speeds up learning without harming solution quality.
4. Genetic and hybrid algorithms - Genetic algorithms encode a packing order and box orientations as chromosomes. Crossover and mutation explore different sequences, while a fitness function rewards higher space utilisation. A 2024 study integrates a Generative Adversarial Network with GA to generate high-quality solutions that improve exploration and exploitation. A 2026 industry-oriented approach (GENPACK) incorporates multiple key performance indicators (KPIs) - space utilisation, support, balance - and uses layer-based chromosomes with specialised crossover operators, achieving up to 35% higher utilisation and improved stability.
5. Multi-stage and reinforcement-learning methods - A two-stage heuristic packs homogeneous product layers using constructive heuristics and then packs leftovers with GA. Deep-reinforcement learning frameworks like O4M-SP use clipped policy gradient methods and reward functions that combine loading rate and height difference; they explicitly handle stability constraints (base support, weight stacking) and generalise across bin sizes. Other RL approaches incorporate partial support and extreme-point sorting, while new work uses transformer-based policies to balance space utilisation with operational time.

These techniques inform the algorithm described below.

## Proposed algorithm: Hybrid extreme-point heuristic with genetic search

### Overview

1. Pre-processing:
   - Rotation generation - For each cuboid, generate the set of valid orientations. If the items are fragile or load-dependent, restrict to allowed rotations.
   - Initial sorting - Sort the cuboids in non-increasing order of volume or by the longest edge. This ordering (similar to First-Fit Decreasing) tends to reduce wasted space.
2. Extreme-point placement heuristic:
   - Initialise an extreme-point list $EP$ with a single point at $(0,0,0)$ (bottom-front-left corner of the container).
   - Maintain a list of empty spaces represented as axis-aligned rectangular parallelepipeds (subspaces) in which boxes may fit. Initially, the entire container is free.
   - For each cuboid in the sorted list:
     1. For each extreme point $p \in EP$ and for each feasible orientation $o$ of the current cuboid:
        - Check whether placing the cuboid at $p$ with orientation $o$ keeps it within the container and does not overlap already placed cuboids. In practice, use a 3-D occupancy grid to perform overlap tests and ensure that partial support conditions are met: the projection of the cuboid's base onto the $xy$-plane must have at least half of its area supported by underlying items or the container floor.
        - Compute a placement score based on two criteria: the unused volume remaining after the placement and the maximum height of the current arrangement. To evaluate a candidate placement, simulate placing the cuboid, update the list of free subspaces, and compute the sum of volumes of these subspaces; this sum represents the unused space inside the container and depends on how the item is positioned. Also compute the resulting maximal height of all placed items.
        - Combine these metrics into a weighted cost, for example $c = \alpha \cdot V_{unused} + \beta \cdot h_{max}$, where $\alpha$ and $\beta$ control the relative importance of wasted space versus height. Lower cost indicates a better placement. If the primary goal is to maximise the number of items, $\alpha$ can be set very high to favour arrangements that leave more space for subsequent items.
     2. Select the placement $(p^{\ast}, o^{\ast})$ with the lowest cost. If multiple placements tie, choose the one that results in greater surface support or better compaction.
     3. Place the cuboid at $(p^{\ast}, o^{\ast})$ and update the occupancy grid.
     4. Update extreme points:
        - Remove $p^{\ast}$ from $EP$.
        - Generate new extreme points at positions created by the faces of the placed cuboid: $(x_i + l_i, y_i, z_i)$, $(x_i, y_i + w_i, z_i)$, and $(x_i, y_i, z_i + h_i)$, where $(x_i, y_i, z_i)$ is the coordinate of the placed cuboid. Only keep points that lie within free space and do not coincide with interior positions of existing items.
        - Optionally apply the Key-Point reduction: prune points that are strictly dominated by another point (i.e., deeper in all three dimensions) to reduce the candidate set.
     5. Update empty spaces by splitting the space occupied by the cuboid from the free subspaces. Add resulting smaller subspaces into the free-space list; remove those fully contained in others.
   - If no feasible placement exists for a cuboid, record a failure (the container is full). Optionally open a new container if multi-bin packing is allowed.

This greedy heuristic provides a fast baseline, but it may leave unused voids due to the fixed item ordering. To further improve utilisation, we integrate a genetic search that explores different sequences and orientations.

## Genetic improvement phase

1. Chromosome representation - Encode a solution as a sequence of item identifiers with associated orientation indices. For instance, chromosome $C = [(i_1, o_1), (i_2, o_2), \ldots]$.
2. Initial population - Generate an initial set of chromosomes. Use the sorted list from the heuristic as one individual. Create others by randomly permuting items and sampling orientations. Optionally include individuals generated by GAN models as suggested by research.
3. Fitness evaluation - For each chromosome, run the extreme-point placement heuristic described above (but without sorting items) and compute:
   - Number of items placed: how many cuboids from the input set fit into the container. Because the goal is to pack as many items as possible, this is the primary objective.
   - Unused volume: the total volume of free subspaces remaining after packing. This metric captures voids both within the stack and above it.
   - Maximum height used and support quality. Solutions using lower height and better contact area are preferred.
   - Combine these metrics into a fitness score that rewards high item counts and low unused volume/height. For example:

     $$
     f = \gamma \cdot n_{placed} - \alpha \cdot V_{unused} - \beta \cdot h_{max} + \delta \cdot s_{support}
     $$

     where $\gamma$ is chosen large enough to prioritise packing more items, and $\alpha$, $\beta$, and $\delta$ adjust the trade-offs between space utilisation, height, and stability. This fitness function ensures that individuals with higher item counts rank above those with lower counts, and within equal counts it prefers arrangements with less wasted space and lower height.
4. Selection - Choose parents via tournament or roulette-wheel selection proportional to fitness.
5. Crossover - Use layer-preserving crossover: split two parent chromosomes at a random position and exchange subsequences while preserving item uniqueness. GENPACK uses a layer-based representation and specialised crossover to maintain feasibility.
6. Mutation - Randomly swap two items, change an orientation, or insert a new extreme-point order. Use a low mutation probability to maintain diversity.
7. Replacement - Form a new population by keeping some elite individuals (elitism) and adding offspring. Repeat evaluation, selection, crossover, and mutation for a fixed number of generations.
8. Termination - Stop when the improvement plateaus or a generation limit is reached. Return the best chromosome.

## Optional reinforcement-learning refinement

After obtaining a good heuristic solution, RL can refine placement decisions in complex scenarios (e.g., irregular item sets, dynamic arrivals). A policy network observes the 3-D occupancy grid state and selects actions (item and placement) among Key Points. The reward function combines space utilisation and stability; partial support constraints are enforced as environment rules. DRL methods like PPO or clipped policy gradient have been used successfully. Training such a model requires simulation but can produce policies that generalise across container sizes and support constraints.

## Pseudocode

The following high-level pseudocode outlines the hybrid algorithm (without GA refinement for brevity). A detailed GA wrapper would call `PlaceItems` for each chromosome.

```python
def pack_cuboids(cuboids, container_dims, alpha=1.0, beta=1.0):
    """Pack as many cuboids as possible into a container while minimising unused volume and height.

    cuboids: list of dictionaries {id, dims: (l, w, h)}
    container_dims: (L, W, H)
    alpha, beta: weights for unused volume and height in the cost function.
    Returns placements for the cuboids that fit; remaining cuboids are unplaced.
    """
    # Precompute all feasible orientations for each cuboid
    for c in cuboids:
        c["orientations"] = unique_orientations(c["dims"])  # six or fewer

    # Sort cuboids in decreasing volume to increase likelihood of packing large items first
    cuboids.sort(key=lambda c: c["dims"][0] * c["dims"][1] * c["dims"][2], reverse=True)

    EP = [(0, 0, 0)]  # list of extreme points
    occupancy = Empty3DGrid(container_dims)  # occupancy grid and free space manager
    placements = {}

    for c in cuboids:
        best_cost = float("inf")
        best_placement = None
        for p in EP:
            for o in c["orientations"]:
                if fits(c, o, p, occupancy, container_dims):
                    # Enforce partial support
                    if not has_partial_support(c, o, p, occupancy):
                        continue
                    # Simulate placing the cuboid at (p, o)
                    sim_occ = occupancy.copy()
                    sim_occ.place(c["id"], o, p)
                    # Compute unused volume after placement
                    unused = sim_occ.total_free_volume()
                    # Compute maximum height after placement
                    height = sim_occ.max_height()
                    # Compute cost: lower unused volume and height are better
                    cost = alpha * unused + beta * height
                    if cost < best_cost:
                        best_cost = cost
                        best_placement = (p, o)
        if best_placement is None:
            # No feasible placement; item cannot be packed
            continue
        # Place the cuboid in the real occupancy grid
        p_star, o_star = best_placement
        occupancy.place(c["id"], o_star, p_star)
        placements[c["id"]] = (p_star, o_star)
        # Update extreme points and free-space list
        EP.remove(p_star)
        new_points = generate_extreme_points(p_star, o_star, c)
        EP.extend(filter_feasible_points(new_points, occupancy, container_dims))
        EP = prune_key_points(EP)
    return placements
```

## Discussion and extensions

This hybrid algorithm combines proven heuristics with metaheuristic search to both maximise the number of items packed and minimise unused space inside the container. The extreme-point list reduces the search space by focusing on promising placements, while partial support and priority sorting encourage stable packings. The placement heuristic now evaluates candidate positions by simulating the effect on unused volume and height and selecting the option that minimises a weighted cost. The genetic search explores multiple sequences, orientations, and item subsets, and its fitness function prioritises packing more items while penalising wasted volume and excessive height. Optional RL refinement can further improve solutions and adapt to dynamic arrival scenarios. Although exact optimality cannot be guaranteed due to NP-hardness, this hybrid approach yields near-optimal packings suitable for applications where both space utilisation and the number of packed items matter.

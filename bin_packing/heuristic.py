from __future__ import annotations

from .exceptions import PackingFailure
from .extreme_points import (
    filter_feasible_points,
    generate_extreme_points,
    generate_projected_points,
    prune_key_points,
)
from .models import Cuboid, Placement, PackingResult
from .occupancy import OccupancyGrid

Point3D = tuple[int, int, int]

SUPPORT_THRESHOLD = 0.5  # at least 50 % of base area must be supported


def fits(
    grid: OccupancyGrid,
    pos: Point3D,
    l: int,
    w: int,
    h: int,
    container_dims: tuple[int, int, int],
) -> bool:
    """True iff placing dims (l, w, h) at pos is within bounds and free."""
    x, y, z = pos
    L, W, H = container_dims
    if x + l > L or y + w > W or z + h > H:
        return False
    return grid.is_free(x, y, z, l, w, h)


def has_partial_support(
    grid: OccupancyGrid,
    pos: Point3D,
    l: int,
    w: int,
) -> bool:
    """True iff >= 50 % of the base footprint is supported (or on the floor)."""
    x, y, z = pos
    return grid.supported_area_fraction(x, y, z, l, w) >= SUPPORT_THRESHOLD


def pack_with_heuristic(
    cuboids: list[Cuboid],
    container_dims: tuple[int, int, int],
    *,
    fail_fast: bool = False,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> PackingResult:
    """
    Run the extreme-point placement heuristic on *cuboids* in the order given.

    Pre-processing (sorting, orientation generation) must be done by the caller.

    Placement scoring: simulate each candidate placement and compute
        cost = alpha * unused_volume + beta * max_height
    selecting the placement with the lowest cost.

    Args:
        cuboids:        Ordered list; each cuboid must have .orientations populated.
        container_dims: (L, W, H) in integer voxel units.
        fail_fast:      Raise PackingFailure on the first unplaceable item when True;
                        otherwise skip the item and record it in unpacked_ids.
        alpha:          Weight for unused volume in placement cost.
        beta:           Weight for max height in placement cost.

    Returns:
        PackingResult with placements and any unplaced items.
    """
    L, W, H = container_dims
    container_vol = L * W * H
    grid = OccupancyGrid(L, W, H)
    ep_list: list[Point3D] = [(0, 0, 0)]
    placements: list[Placement] = []
    placed_extents: list[tuple[int, int, int, int, int, int]] = []
    unpacked_ids: list[str] = []

    h_max_so_far: int = 0
    V_packed_so_far: int = 0

    for cuboid in cuboids:
        best_cost: float = float("inf")
        best_pos: Point3D | None = None
        best_orient = None

        for pos in ep_list:
            for orient in cuboid.orientations:
                ol, ow, oh = int(orient.l), int(orient.w), int(orient.h)
                if not fits(grid, pos, ol, ow, oh, container_dims):
                    continue
                if not has_partial_support(grid, pos, ol, ow):
                    continue
                x, y, z = pos
                h_max = max(h_max_so_far, z + oh)
                cost = alpha * (container_vol - V_packed_so_far - ol * ow * oh) + beta * h_max
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
                    best_orient = orient

        if best_pos is None:
            if fail_fast:
                raise PackingFailure(f"Cannot place item {cuboid.item_id!r}")
            unpacked_ids.append(cuboid.item_id)
            continue

        x, y, z = best_pos
        ol, ow, oh = int(best_orient.l), int(best_orient.w), int(best_orient.h)

        grid.place(cuboid.item_id, x, y, z, ol, ow, oh)
        placements.append(
            Placement(item_id=cuboid.item_id, orientation=best_orient, position=best_pos)
        )
        h_max_so_far   = max(h_max_so_far, z + oh)
        V_packed_so_far += ol * ow * oh

        # Update EP list
        ep_list.remove(best_pos)

        # Standard EPs from the 3 faces of the newly placed box
        new_pts = generate_extreme_points(best_pos, best_orient)

        # Projected EPs against all previously placed boxes (before appending current)
        new_pts.extend(generate_projected_points(best_pos, best_orient, placed_extents))
        placed_extents.append((x, y, z, ol, ow, oh))

        # Filter to in-bounds, unoccupied corner voxels; deduplicate before extending
        filtered = filter_feasible_points(new_pts, grid, container_dims)
        existing_set = set(ep_list)
        for pt in filtered:
            if pt not in existing_set:
                ep_list.append(pt)
                existing_set.add(pt)

        # Occupancy-verified dominance pruning
        ep_list = prune_key_points(ep_list, grid)

    return PackingResult(
        placements=placements,
        unpacked_ids=unpacked_ids,
        container_dims=container_dims,
    )

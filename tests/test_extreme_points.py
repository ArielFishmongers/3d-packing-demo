import pytest
from bin_packing.models import Orientation, Placement
from bin_packing.occupancy import OccupancyGrid
from bin_packing.extreme_points import (
    generate_extreme_points,
    filter_feasible_points,
    rebuild_key_points,
    rebuild_key_points_from_axes,
    prune_key_points,
)


@pytest.fixture
def container():
    return (10, 10, 10)


@pytest.fixture
def empty_grid():
    return OccupancyGrid(10, 10, 10)


def test_generate_three_eps():
    pos = (0, 0, 0)
    o = Orientation(l=3, w=4, h=5)
    eps = generate_extreme_points(pos, o)
    assert len(eps) == 3
    assert (3, 0, 0) in eps
    assert (0, 4, 0) in eps
    assert (0, 0, 5) in eps


def test_generate_eps_mid_position():
    pos = (2, 3, 1)
    o = Orientation(l=2, w=2, h=2)
    eps = generate_extreme_points(pos, o)
    assert (4, 3, 1) in eps
    assert (2, 5, 1) in eps
    assert (2, 3, 3) in eps


def test_filter_removes_out_of_bounds(empty_grid, container):
    pts = [(10, 0, 0), (0, 10, 0), (0, 0, 10), (9, 9, 9)]
    result = filter_feasible_points(pts, empty_grid, container)
    assert result == [(9, 9, 9)]


def test_filter_keeps_all_feasible(empty_grid, container):
    pts = [(0, 0, 0), (1, 2, 3), (9, 9, 9)]
    result = filter_feasible_points(pts, empty_grid, container)
    assert set(result) == {(0, 0, 0), (1, 2, 3), (9, 9, 9)}


def test_filter_removes_occupied():
    grid = OccupancyGrid(10, 10, 10)
    grid.place("a", 3, 0, 0, 1, 1, 1)
    pts = [(3, 0, 0), (5, 5, 5)]
    result = filter_feasible_points(pts, grid, (10, 10, 10))
    assert (3, 0, 0) not in result
    assert (5, 5, 5) in result


def test_filter_negative_coords(empty_grid, container):
    pts = [(-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    result = filter_feasible_points(pts, empty_grid, container)
    assert result == []


def test_prune_removes_dominated(empty_grid):
    # (2,2,2) is dominated by (1,1,1) and the gap is empty
    eps = [(1, 1, 1), (2, 2, 2)]
    result = prune_key_points(eps, empty_grid)
    assert (1, 1, 1) in result
    assert (2, 2, 2) not in result


def test_prune_keeps_non_dominated(empty_grid):
    # Neither dominates the other: (1,5,1) vs (5,1,1)
    eps = [(1, 5, 1), (5, 1, 1)]
    result = prune_key_points(eps, empty_grid)
    assert len(result) == 2


def test_prune_single_point(empty_grid):
    eps = [(3, 3, 3)]
    result = prune_key_points(eps, empty_grid)
    assert result == [(3, 3, 3)]


def test_prune_empty(empty_grid):
    assert prune_key_points([], empty_grid) == []


def test_prune_idempotent(empty_grid):
    eps = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    first = prune_key_points(eps, empty_grid)
    second = prune_key_points(first, empty_grid)
    assert set(first) == set(second)


def test_prune_does_not_remove_equal_points(empty_grid):
    # Two identical points — neither strictly dominates the other (q != p check)
    eps = [(2, 2, 2), (2, 2, 2)]
    result = prune_key_points(eps, empty_grid)
    # Both should survive (they're equal, not dominated)
    assert len(result) == 2


def test_prune_keeps_blocked_dominated_point():
    # (2,2,2) is geometrically dominated by (1,1,1), but there is a box
    # in the gap between them — so (2,2,2) must NOT be pruned.
    grid = OccupancyGrid(10, 10, 10)
    grid.place("blocker", 1, 1, 1, 1, 1, 1)   # occupies exactly the gap voxel
    eps = [(1, 1, 1), (2, 2, 2)]
    result = prune_key_points(eps, grid)
    assert (1, 1, 1) in result
    assert (2, 2, 2) in result  # must be kept — gap is blocked


def test_prune_keeps_point_with_obstacle_outside_gap():
    # Q=(0,0,5), P=(5,3,5): Q ≤ P component-wise, gap (0..4, 0..2, 5) is free.
    # But an obstacle sits at (2, 4, 5) — outside the gap's y-range.
    # A future box at Q extending to y=4 would collide; P is the safe position.
    # The fix: partial-zero-delta cases (here dz=0) must NOT prune P.
    grid = OccupancyGrid(20, 20, 20)
    grid.place("obstacle", 2, 4, 5, 3, 1, 5)  # (2..4, 4, 5..9) — outside gap
    eps = [(0, 0, 5), (5, 3, 5)]
    result = prune_key_points(eps, grid)
    assert (0, 0, 5) in result
    assert (5, 3, 5) in result  # must be kept — obstacle outside gap is missed


def test_prune_below_dominance_still_works():
    # Q=(3,3,0) directly below P=(3,3,5) — column between them is free.
    # This safe "directly below" case SHOULD still prune P.
    grid = OccupancyGrid(10, 10, 10)
    eps = [(3, 3, 0), (3, 3, 5)]
    result = prune_key_points(eps, grid)
    assert (3, 3, 0) in result
    assert (3, 3, 5) not in result  # correctly dominated — lower is always better


def test_rebuild_key_points_restores_older_neighbor_floor_anchor():
    container = (8, 8, 4)
    grid = OccupancyGrid(*container)
    placements = [
        Placement("a", Orientation(l=1, w=3, h=1), (0, 0, 0)),
        Placement("b", Orientation(l=2, w=2, h=1), (1, 0, 0)),
        Placement("c", Orientation(l=3, w=1, h=1), (3, 0, 0)),
    ]
    for placement in placements:
        grid.place(
            placement.item_id,
            int(placement.position[0]),
            int(placement.position[1]),
            int(placement.position[2]),
            int(placement.orientation.l),
            int(placement.orientation.w),
            int(placement.orientation.h),
        )

    key_points = rebuild_key_points(placements, grid, container)

    assert (1, 3, 0) in key_points


def test_rebuild_key_points_from_axes_matches_placement_rebuild():
    container = (8, 8, 4)
    grid = OccupancyGrid(*container)
    placements = [
        Placement("a", Orientation(l=1, w=3, h=1), (0, 0, 0)),
        Placement("b", Orientation(l=2, w=2, h=1), (1, 0, 0)),
        Placement("c", Orientation(l=3, w=1, h=1), (3, 0, 0)),
    ]
    xs = {0}
    ys = {0}
    zs = {0}
    for placement in placements:
        grid.place(
            placement.item_id,
            int(placement.position[0]),
            int(placement.position[1]),
            int(placement.position[2]),
            int(placement.orientation.l),
            int(placement.orientation.w),
            int(placement.orientation.h),
        )
        xs.add(int(placement.x_max))
        ys.add(int(placement.y_max))
        zs.add(int(placement.z_max))

    from_placements = rebuild_key_points(placements, grid, container)
    from_axes = rebuild_key_points_from_axes(xs, ys, zs, grid, container)

    assert from_axes == from_placements

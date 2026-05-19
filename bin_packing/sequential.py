from __future__ import annotations

from .extreme_points import (
    rebuild_key_points_from_axes,
)
from .heuristic import fits, has_partial_support
from .models import Cuboid, Orientation, PackingResult, Placement
from .occupancy import OccupancyGrid
from .orientations import unique_orientations

Point3D = tuple[int, int, int]


def _orientation_sort_key(orientation: Orientation) -> tuple[int, int, tuple[int, int, int]]:
    dims = (int(orientation.l), int(orientation.w), int(orientation.h))
    return (-dims[0] * dims[1], dims[2], dims)


def _ensure_orientations(cuboids: list[Cuboid]) -> None:
    """Populate and order orientations for sequential evaluation."""
    for c in cuboids:
        if not c.orientations:
            if c.rotations_allowed:
                c.orientations = unique_orientations(c.dims)
            else:
                c.orientations = [Orientation(l=c.dims[0], w=c.dims[1], h=c.dims[2])]
        c.orientations = sorted(c.orientations, key=_orientation_sort_key)


def _support_fraction(
    grid: OccupancyGrid,
    pos: Point3D,
    l: int,
    w: int,
) -> float:
    x, y, z = pos
    if z == 0:
        return 1.0
    return grid.supported_area_fraction(x, y, z, l, w)


def _is_feasible_candidate(
    grid: OccupancyGrid,
    pos: Point3D,
    orientation: Orientation,
    container_dims: tuple[int, int, int],
) -> bool:
    l, w, h = int(orientation.l), int(orientation.w), int(orientation.h)
    if not fits(grid, pos, l, w, h, container_dims):
        return False
    if pos[2] > 0 and not has_partial_support(grid, pos, l, w):
        return False
    return True


def pack_sequential(
    cuboids: list[Cuboid],
    container_dims: tuple[int, int, int],
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> PackingResult:
    """
    Pack cuboids in strict arrival order (no sorting, no skipping).

    Candidate anchors are rebuilt after each placement from the occupied face
    coordinates of all committed boxes. Selection is greedy and uses only the
    current state of the container: choose from the lowest feasible layer, then
    rank candidates by weighted delta_h, weighted delta_xy, weighted V_void,
    higher support fraction, lower y/x, then orientation rank.

    On the first cuboid that cannot be placed, the algorithm stops
    immediately. That cuboid and all remaining cuboids are recorded in
    unpacked_ids (in arrival order).

    Args:
        cuboids:        Items in the order they must be packed. Each
                        cuboid's .orientations field is populated in-place
                        if not already set; the list itself is not sorted.
        container_dims: (L, W, H) in integer voxel units.
        alpha:          Weight for XY-growth and void comparisons.
        beta:           Weight for height-growth comparisons.

    Returns:
        PackingResult with the placed prefix and any unpacked suffix.
    """
    _ensure_orientations(cuboids)

    L, W, H = container_dims
    grid = OccupancyGrid(L, W, H)
    key_points: list[Point3D] = [(0, 0, 0)]
    placements: list[Placement] = []
    unpacked_ids: list[str] = []

    x_max_so_far: int = 0
    y_max_so_far: int = 0
    h_max_so_far: int = 0
    packed_volume_so_far: int = 0
    xs = {0}
    ys = {0}
    zs = {0}

    for i, cuboid in enumerate(cuboids):
        best_z: int | None = None
        best_key: tuple | None = None
        best_pos: Point3D | None = None
        best_orient: Orientation | None = None
        current_area = x_max_so_far * y_max_so_far

        for ori_idx, orient in enumerate(cuboid.orientations):
            ol, ow, oh = int(orient.l), int(orient.w), int(orient.h)

            for pos in key_points:
                xp, yp, zp = pos

                if not _is_feasible_candidate(grid, pos, orient, container_dims):
                    continue

                if best_z is not None and zp > best_z:
                    continue

                x_max = max(x_max_so_far, xp + ol)
                y_max = max(y_max_so_far, yp + ow)
                h_max = max(h_max_so_far, zp + oh)
                packed_volume = packed_volume_so_far + (ol * ow * oh)
                delta_h = max(0, h_max - h_max_so_far)
                delta_xy = (x_max * y_max) - current_area
                V_void = x_max * y_max * h_max - packed_volume
                support_frac = _support_fraction(grid, pos, ol, ow)

                key = (
                    beta * delta_h,
                    alpha * delta_xy,
                    alpha * V_void,
                    -support_frac,
                    yp,
                    xp,
                    ori_idx,
                )

                if best_z is None or zp < best_z:
                    best_z = zp
                    best_key = key
                    best_pos = pos
                    best_orient = orient
                    continue

                if key < best_key:
                    best_key = key
                    best_pos = pos
                    best_orient = orient

        # --- Stop-on-failure ---
        if best_pos is None:
            unpacked_ids.extend(c.item_id for c in cuboids[i:])
            break

        # --- Commit best placement ---
        xp, yp, zp = best_pos
        ol, ow, oh = int(best_orient.l), int(best_orient.w), int(best_orient.h)

        grid.place(cuboid.item_id, xp, yp, zp, ol, ow, oh)
        placements.append(
            Placement(item_id=cuboid.item_id, orientation=best_orient, position=best_pos)
        )

        x_max_so_far = max(x_max_so_far, xp + ol)
        y_max_so_far = max(y_max_so_far, yp + ow)
        h_max_so_far = max(h_max_so_far, zp + oh)
        packed_volume_so_far += ol * ow * oh
        xs.add(xp + ol)
        ys.add(yp + ow)
        zs.add(zp + oh)
        key_points = rebuild_key_points_from_axes(xs, ys, zs, grid, container_dims)

    return PackingResult(
        placements=placements,
        unpacked_ids=unpacked_ids,
        container_dims=container_dims,
    )

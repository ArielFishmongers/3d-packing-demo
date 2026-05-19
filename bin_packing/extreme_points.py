from __future__ import annotations

from .models import Orientation, Placement
from .occupancy import OccupancyGrid

Point3D = tuple[int, int, int]
PlacedExtent = tuple[int, int, int, int, int, int]  # (x, y, z, l, w, h)


def generate_extreme_points(pos: Point3D, orientation: Orientation) -> list[Point3D]:
    """
    Generate the 3 new candidate extreme points from the faces of a placed cuboid.

    pos = (x, y, z) is the bottom-front-left corner.
    The new EPs are at:
      - (x + l, y,     z)     — right face
      - (x,     y + w, z)     — back face
      - (x,     y,     z + h) — top face
    """
    x, y, z = pos
    l, w, h = int(orientation.l), int(orientation.w), int(orientation.h)
    return [
        (x + l, y, z),
        (x, y + w, z),
        (x, y, z + h),
    ]


def generate_projected_points(
    pos: Point3D,
    orientation: Orientation,
    placed_extents: list[PlacedExtent],
) -> list[Point3D]:
    """
    Project each face of the newly placed box against each existing box's edges
    to generate additional extreme-point candidates.

    For each face of the new box and each existing box E at (ex, ey, ez) with
    dims (el, ew, eh), combine the face coordinate with E's edge coordinates
    for the other two axes:

      right face (x = px+ol) → (px+ol, ey, ez), (px+ol, ey+ew, ez), (px+ol, ey, ez+eh)
      back  face (y = py+ow) → (ex, py+ow, ez), (ex+el, py+ow, ez), (ex, py+ow, ez+eh)
      top   face (z = pz+oh) → (ex, ey, pz+oh), (ex+el, ey, pz+oh), (ex, ey+ew, pz+oh)

    Returns up to 9 × len(placed_extents) candidate points before filtering.
    """
    px, py, pz = pos
    ol, ow, oh = int(orientation.l), int(orientation.w), int(orientation.h)
    pts: list[Point3D] = []
    for ex, ey, ez, el, ew, eh in placed_extents:
        # Project new box's right face against existing box edges
        pts += [
            (px + ol, ey,      ez     ),
            (px + ol, ey + ew, ez     ),
            (px + ol, ey,      ez + eh),
        ]
        # Project new box's back face against existing box edges
        pts += [
            (ex,      py + ow, ez     ),
            (ex + el, py + ow, ez     ),
            (ex,      py + ow, ez + eh),
        ]
        # Project new box's top face against existing box edges
        pts += [
            (ex,      ey,      pz + oh),
            (ex + el, ey,      pz + oh),
            (ex,      ey + ew, pz + oh),
        ]
    return pts


def filter_feasible_points(
    points: list[Point3D],
    grid: OccupancyGrid,
    container_dims: tuple[int, int, int],
) -> list[Point3D]:
    """
    Keep only extreme points that are:
      1. Within container bounds (strictly: x < L, y < W, z < H).
      2. Not already occupied.
    """
    L, W, H = container_dims
    result: list[Point3D] = []
    for x, y, z in points:
        if 0 <= x < L and 0 <= y < W and 0 <= z < H:
            if grid._grid[x, y, z] == 0:
                result.append((x, y, z))
    return result


def rebuild_key_points(
    placements: list[Placement],
    grid: OccupancyGrid,
    container_dims: tuple[int, int, int],
) -> list[Point3D]:
    """
    Rebuild the candidate anchor set from all committed placement faces.

    Sequential packing uses this global rebuild instead of incremental EP
    growth so that anchors formed by combinations of older boxes are not lost.
    """
    if not placements:
        return [(0, 0, 0)]

    xs = {0}
    ys = {0}
    zs = {0}
    for placement in placements:
        xs.add(int(placement.x_max))
        ys.add(int(placement.y_max))
        zs.add(int(placement.z_max))

    return rebuild_key_points_from_axes(xs, ys, zs, grid, container_dims)


def rebuild_key_points_from_axes(
    xs: set[int],
    ys: set[int],
    zs: set[int],
    grid: OccupancyGrid,
    container_dims: tuple[int, int, int],
) -> list[Point3D]:
    """Rebuild key points from precomputed committed face coordinates."""
    if not xs or not ys or not zs:
        return [(0, 0, 0)]

    points = [
        (x, y, z)
        for x in sorted(xs)
        for y in sorted(ys)
        for z in sorted(zs)
    ]
    feasible_points = filter_feasible_points(points, grid, container_dims)
    return prune_key_points(feasible_points, grid)


def prune_key_points(
    ep_list: list[Point3D],
    grid: OccupancyGrid,
) -> list[Point3D]:
    """
    Remove dominated extreme points, verified against the current occupancy.

    Point Q dominates point P iff:
      Q.x <= P.x  AND  Q.y <= P.y  AND  Q.z <= P.z  AND  Q != P
      AND the bounding region from Q to P is completely empty in the grid.

    The occupancy check ensures we only prune P when any box fitting at P
    would genuinely also fit at Q — i.e. there are no obstacles in the gap
    between them that could block Q's footprint even if P's is clear.

    O(n² × V_gap) — the EP list stays small so this is fast in practice.
    """
    pruned: list[Point3D] = []
    for i, p in enumerate(ep_list):
        dominated = False
        for j, q in enumerate(ep_list):
            if i == j:
                continue
            if q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2] and q != p:
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                dz = p[2] - q[2]
                if dx > 0 and dy > 0 and dz > 0:
                    # Full 3D gap — safe to check the exact bounding box.
                    if grid.is_free(q[0], q[1], q[2], dx, dy, dz):
                        dominated = True
                        break
                elif dx == 0 and dy == 0 and dz > 0:
                    # Q is directly below P on the same XY position.
                    # Safe: any box at height P can also start at height Q if
                    # the vertical column between them is clear.
                    if grid.is_free(q[0], q[1], q[2], 1, 1, dz):
                        dominated = True
                        break
                # All other partial-zero cases (same x-plane, same y-plane,
                # same z-plane with 2-D neighbours) are skipped: the gap check
                # would only cover a thin slice and can miss obstacles that lie
                # outside that slice but still within the reach of a future box
                # placed at Q.  Keeping P here is conservative but correct.
        if not dominated:
            pruned.append(p)
    return pruned

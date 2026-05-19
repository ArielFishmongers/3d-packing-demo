from itertools import permutations

from .models import Cuboid, Orientation


def unique_orientations(dims: tuple[float, float, float]) -> list[Orientation]:
    """
    Return the deduplicated list of axis-aligned orientations for a cuboid.

    A generic box (a, b, c) with all distinct dims has 6 orientations.
    A square prism (a, a, b) has 3. A cube (a, a, a) has 1.
    """
    seen: set[tuple[float, float, float]] = set()
    result: list[Orientation] = []
    for perm in permutations(dims):
        if perm not in seen:
            seen.add(perm)
            result.append(Orientation(l=perm[0], w=perm[1], h=perm[2]))
    return result


def preprocess_cuboids(
    cuboids: list[Cuboid],
    sort_by: str = "volume",
) -> list[Cuboid]:
    """
    Populate each cuboid's orientations list in-place, then return a new
    list sorted in decreasing order of the chosen key.

    Args:
        cuboids: input list (mutated in-place for orientations).
        sort_by: "volume" (default) or "longest_edge".
    """
    for c in cuboids:
        if c.rotations_allowed:
            c.orientations = unique_orientations(c.dims)
        else:
            c.orientations = [Orientation(l=c.dims[0], w=c.dims[1], h=c.dims[2])]

    if sort_by == "volume":
        return sorted(cuboids, key=lambda c: c.volume, reverse=True)
    elif sort_by == "longest_edge":
        return sorted(cuboids, key=lambda c: max(c.dims), reverse=True)
    else:
        raise ValueError(f"Unknown sort_by: {sort_by!r}. Use 'volume' or 'longest_edge'.")

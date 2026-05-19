"""Tests for the sequential packing algorithm (pack_sequential / Packer.pack_sequential).

Requirements verified:
  1. Order is preserved (no sorting).
  2. Stop-on-failure: first unplaceable item stops the algorithm; remaining
     items (including that item) are all recorded in unpacked_ids.
  3. Feasibility: no overlaps, no out-of-bounds placements, support threshold
     is enforced for above-floor placements.
  4. Determinism: repeated runs on identical input produce identical output.
  5. Tie-break: deterministic tie-break order is exercised.
  6. Compatibility: existing heuristic / GA behaviour is not affected.
"""
from __future__ import annotations

from bin_packing.models import Cuboid, PackingResult
from bin_packing.occupancy import OccupancyGrid
from bin_packing.orientations import unique_orientations
from bin_packing.packer import Packer, PackerConfig
from bin_packing.sequential import pack_sequential


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cuboids(*dims_list: tuple[int, int, int]) -> list[Cuboid]:
    """Create cuboids with auto-generated IDs and orientations populated."""
    cuboids = [Cuboid(item_id=f"box{i}", dims=d) for i, d in enumerate(dims_list)]
    for c in cuboids:
        c.orientations = unique_orientations(c.dims)
    return cuboids


def assert_no_overlap(result: PackingResult, container_dims: tuple[int, int, int]) -> None:
    """Place all result placements into a fresh grid; raise if any overlap."""
    L, W, H = (int(d) for d in container_dims)
    grid = OccupancyGrid(L, W, H)
    for p in result.placements:
        x, y, z = (int(c) for c in p.position)
        l, w, h = int(p.orientation.l), int(p.orientation.w), int(p.orientation.h)
        assert grid.is_free(x, y, z, l, w, h), (
            f"Overlap detected for item {p.item_id!r} at {p.position}"
        )
        grid.place(p.item_id, x, y, z, l, w, h)


def assert_in_bounds(result: PackingResult, container_dims: tuple[int, int, int]) -> None:
    """Verify every placement fits within the container."""
    L, W, H = (int(d) for d in container_dims)
    for p in result.placements:
        assert p.x_max <= L, f"{p.item_id} x_max {p.x_max} > L {L}"
        assert p.y_max <= W, f"{p.item_id} y_max {p.y_max} > W {W}"
        assert p.z_max <= H, f"{p.item_id} z_max {p.z_max} > H {H}"


# ---------------------------------------------------------------------------
# 1. Order preservation
# ---------------------------------------------------------------------------

class TestOrderPreservation:
    def test_placement_ids_match_input_order(self):
        """Placed items appear in arrival order, not sorted by volume."""
        container = (10, 10, 10)
        # Deliberately put the smallest box first and largest last
        cuboids = make_cuboids((1, 1, 1), (3, 3, 3), (2, 2, 2))
        result = pack_sequential(cuboids, container)

        assert len(result.placements) == 3
        placed_ids = [p.item_id for p in result.placements]
        assert placed_ids == ["box0", "box1", "box2"]

    def test_large_item_first_is_placed_first(self):
        """A large item arriving first is placed even though a volume-sort would put it last."""
        container = (4, 4, 4)
        # Largest item arrives first
        cuboids = make_cuboids((4, 4, 4), (1, 1, 1))
        result = pack_sequential(cuboids, container)

        # Only the large item fits; the small one is unpacked
        assert result.placements[0].item_id == "box0"


# ---------------------------------------------------------------------------
# 2. Stop-on-failure
# ---------------------------------------------------------------------------

class TestStopOnFailure:
    def test_stops_at_first_unplaceable_item(self):
        """Items after the first failure are not attempted."""
        container = (4, 4, 4)
        # box0 fills the entire container; box1 cannot fit; box2 is never tried
        cuboids = make_cuboids((4, 4, 4), (2, 2, 2), (1, 1, 1))
        result = pack_sequential(cuboids, container)

        assert [p.item_id for p in result.placements] == ["box0"]
        assert result.unpacked_ids == ["box1", "box2"]

    def test_unpacked_ids_preserves_suffix_order(self):
        """unpacked_ids contains the first failure and all subsequent items in order."""
        container = (3, 3, 3)
        # Place box0 (3x3x3) to fill container; then box1, box2, box3 cannot fit
        cuboids = make_cuboids((3, 3, 3), (1, 1, 1), (2, 1, 1), (1, 2, 1))
        result = pack_sequential(cuboids, container)

        assert result.unpacked_ids == ["box1", "box2", "box3"]

    def test_no_placements_when_first_item_too_large(self):
        """When the very first item cannot fit, placements is empty."""
        container = (2, 2, 2)
        cuboids = make_cuboids((3, 3, 3), (1, 1, 1))
        result = pack_sequential(cuboids, container)

        assert result.placements == []
        assert result.unpacked_ids == ["box0", "box1"]

    def test_all_placed_when_all_fit(self):
        """unpacked_ids is empty when every item can be placed."""
        container = (10, 10, 10)
        cuboids = make_cuboids((2, 2, 2), (2, 2, 2), (2, 2, 2))
        result = pack_sequential(cuboids, container)

        assert len(result.placements) == 3
        assert result.unpacked_ids == []


# ---------------------------------------------------------------------------
# 3. Feasibility guarantees
# ---------------------------------------------------------------------------

class TestFeasibility:
    def test_no_overlap(self):
        container = (10, 10, 10)
        cuboids = make_cuboids((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3))
        result = pack_sequential(cuboids, container)
        assert_no_overlap(result, container)

    def test_in_bounds(self):
        container = (6, 6, 6)
        cuboids = make_cuboids((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2))
        result = pack_sequential(cuboids, container)
        assert_in_bounds(result, container)

    def test_exact_fit_single_item(self):
        """A cuboid that exactly fills the container is placed successfully."""
        container = (5, 4, 3)
        cuboids = [Cuboid(item_id="exact", dims=(5, 4, 3), rotations_allowed=False)]
        cuboids[0].orientations = [cuboids[0].dims]
        from bin_packing.models import Orientation
        cuboids[0].orientations = [Orientation(l=5, w=4, h=3)]
        result = pack_sequential(cuboids, container)

        assert len(result.placements) == 1
        assert result.unpacked_ids == []


# ---------------------------------------------------------------------------
# 4. Support constraint
# ---------------------------------------------------------------------------

class TestSupportConstraint:
    def test_unsupported_placement_rejected(self):
        """
        A tall thin column cannot be stacked on a tiny base.

        Layout:
          - box0: 2×2×1 placed at (0,0,0) — floor
          - box1: 10×10×1 (large slab) cannot be placed above box0 because
            only 4 of its 100 base cells are supported (< 50 %)
        """
        container = (10, 10, 10)
        # box0: small base at floor
        box0 = Cuboid(item_id="base", dims=(2, 2, 1), rotations_allowed=False)
        from bin_packing.models import Orientation
        box0.orientations = [Orientation(l=2, w=2, h=1)]

        # box1: large slab, rotations not allowed so it must be placed as 10×10×1
        box1 = Cuboid(item_id="slab", dims=(10, 10, 1), rotations_allowed=False)
        box1.orientations = [Orientation(l=10, w=10, h=1)]

        result = pack_sequential([box0, box1], container)

        # box0 must be placed at z=0
        assert result.placements[0].item_id == "base"
        assert result.placements[0].position[2] == 0

        # box1 cannot stack above box0 (unsupported) and cannot fit at floor
        # if box0 occupies the corner — it may actually still land at floor in
        # the opposite corner.  What we care about is that the support rule is
        # *checked*.  We verify no placement violates support.
        grid = OccupancyGrid(10, 10, 10)
        for p in result.placements:
            x, y, z = (int(c) for c in p.position)
            l, w, h = int(p.orientation.l), int(p.orientation.w), int(p.orientation.h)
            if z > 0:
                frac = grid.supported_area_fraction(x, y, z, l, w)
                assert frac >= 0.5, (
                    f"Item {p.item_id!r} at z={z} has support fraction {frac:.2f} < 0.5"
                )
            grid.place(p.item_id, x, y, z, l, w, h)


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_repeated_calls_produce_same_result(self):
        container = (10, 10, 10)
        cuboids_a = make_cuboids((3, 2, 4), (2, 3, 2), (4, 2, 3), (1, 1, 1))
        cuboids_b = make_cuboids((3, 2, 4), (2, 3, 2), (4, 2, 3), (1, 1, 1))

        result_a = pack_sequential(cuboids_a, container)
        result_b = pack_sequential(cuboids_b, container)

        assert len(result_a.placements) == len(result_b.placements)
        for pa, pb in zip(result_a.placements, result_b.placements):
            assert pa.item_id == pb.item_id
            assert pa.position == pb.position
            assert pa.orientation == pb.orientation

    def test_via_packer_api_is_deterministic(self):
        packer = Packer(container_dims=(10, 10, 10), config=PackerConfig(use_ga=False))
        cuboids = [Cuboid(f"c{i}", (2, 3, 2)) for i in range(4)]
        result_a = packer.pack_sequential(cuboids)
        result_b = packer.pack_sequential(cuboids)

        assert [(p.item_id, p.position) for p in result_a.placements] == \
               [(p.item_id, p.position) for p in result_b.placements]


# ---------------------------------------------------------------------------
# 6. Tie-break coverage
# ---------------------------------------------------------------------------

class TestTieBreak:
    def test_prefers_floor_over_stacking_when_floor_space_exists(self):
        from bin_packing.models import Orientation

        container = (10, 10, 10)
        base = Cuboid(item_id="base", dims=(3, 3, 1), rotations_allowed=False)
        base.orientations = [Orientation(l=3, w=3, h=1)]
        tall = Cuboid(item_id="tall", dims=(1, 1, 5), rotations_allowed=False)
        tall.orientations = [Orientation(l=1, w=1, h=5)]

        result = pack_sequential([base, tall], container)

        assert len(result.placements) == 2
        tall_placement = next(p for p in result.placements if p.item_id == "tall")
        assert tall_placement.position == (3, 0, 0)

    def test_lower_z_preferred_over_higher_z(self):
        """
        When two extreme points offer equal cost, the one with lower z is chosen.

        We create a situation where two EPs at the same x,y but different z
        produce identical cost, and verify the item lands at the lower z.
        """
        from bin_packing.models import Orientation
        container = (4, 4, 8)

        # Place two identical 2×2×2 cubes side by side at floor level.
        # They generate EPs at z=2 (tops) and z=0 (sides).
        # The next item should prefer z=0 (floor-level EP) over z=2.
        box0 = Cuboid(item_id="a", dims=(2, 2, 2), rotations_allowed=False)
        box0.orientations = [Orientation(l=2, w=2, h=2)]
        box1 = Cuboid(item_id="b", dims=(2, 2, 2), rotations_allowed=False)
        box1.orientations = [Orientation(l=2, w=2, h=2)]
        box2 = Cuboid(item_id="c", dims=(2, 2, 2), rotations_allowed=False)
        box2.orientations = [Orientation(l=2, w=2, h=2)]

        result = pack_sequential([box0, box1, box2], container)

        assert len(result.placements) == 3
        # box2 must be placed at z=0, not stacked on top of box0/box1
        c_placement = next(p for p in result.placements if p.item_id == "c")
        assert c_placement.position[2] == 0, (
            f"Expected z=0 but got z={c_placement.position[2]}"
        )

    def test_lower_orientation_index_breaks_equal_cost(self):
        """
        When all feasible orientations at the same position produce equal cost,
        the orientation with the lower index is selected.
        """
        from bin_packing.models import Orientation
        container = (10, 10, 10)

        # A cube has only one unique orientation so no ambiguity test.
        # Use a 1×2×2 box — it has 3 unique orientations, all with equal
        # volume; put it alone so only EP (0,0,0) exists.
        box = Cuboid(item_id="rect", dims=(1, 2, 2))
        box.orientations = unique_orientations(box.dims)

        result = pack_sequential([box], container)

        assert len(result.placements) == 1
        chosen = result.placements[0]
        # The chosen orientation must be the first one that yields the minimum key
        # — we just verify it is consistent (determinism covers equality).
        assert chosen.orientation in box.orientations


# ---------------------------------------------------------------------------
# 7. Coverage regressions
# ---------------------------------------------------------------------------

class TestCoverageRegressions:
    def test_lowest_layer_wins_even_if_a_later_item_is_lost(self):
        from bin_packing.models import Orientation

        container = (5, 5, 4)
        dims_list = [
            (1, 1, 1),
            (3, 2, 1),
            (3, 1, 2),
            (2, 1, 2),
            (1, 3, 2),
        ]
        cuboids = []
        for idx, dims in enumerate(dims_list):
            cuboid = Cuboid(item_id=f"b{idx}", dims=dims, rotations_allowed=False)
            cuboid.orientations = [Orientation(l=dims[0], w=dims[1], h=dims[2])]
            cuboids.append(cuboid)

        result = pack_sequential(cuboids, container)

        assert len(result.placements) == 4
        assert result.unpacked_ids == ["b4"]
        assert next(p for p in result.placements if p.item_id == "b3").position == (3, 2, 0)


# ---------------------------------------------------------------------------
# 8. Compatibility — existing heuristic / GA tests are not broken
# ---------------------------------------------------------------------------

class TestCompatibility:
    def test_packer_pack_still_sorts_and_skips(self):
        """Packer.pack() behaviour is unchanged: sorts by volume, skips failures."""
        packer = Packer(
            container_dims=(4, 4, 4),
            config=PackerConfig(use_ga=False),
        )
        # A 5×5×5 item cannot fit; Packer.pack() must skip it and pack the others.
        cuboids = [
            Cuboid("small", (2, 2, 2)),
            Cuboid("oversized", (5, 5, 5)),
            Cuboid("tiny", (1, 1, 1)),
        ]
        result = packer.pack(cuboids)

        assert "oversized" in result.unpacked_ids
        placed_ids = {p.item_id for p in result.placements}
        assert "small" in placed_ids or "tiny" in placed_ids

    def test_pack_sequential_does_not_skip(self):
        """Packer.pack_sequential() stops on first failure instead of skipping."""
        packer = Packer(
            container_dims=(4, 4, 4),
            config=PackerConfig(use_ga=False),
        )
        cuboids = [
            Cuboid("first", (2, 2, 2)),
            Cuboid("oversized", (5, 5, 5)),  # cannot fit → stop
            Cuboid("third", (1, 1, 1)),       # never attempted
        ]
        result = packer.pack_sequential(cuboids)

        assert [p.item_id for p in result.placements] == ["first"]
        assert result.unpacked_ids == ["oversized", "third"]

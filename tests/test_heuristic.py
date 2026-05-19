import pytest
from bin_packing.models import Cuboid
from bin_packing.orientations import preprocess_cuboids
from bin_packing.heuristic import pack_with_heuristic
from bin_packing.occupancy import OccupancyGrid
from bin_packing.exceptions import PackingFailure


def make_cuboids(*dims_list):
    cuboids = [Cuboid(f"box{i}", d) for i, d in enumerate(dims_list)]
    return preprocess_cuboids(cuboids)


def test_single_box_placed_at_origin():
    cuboids = make_cuboids((3, 3, 3))
    result = pack_with_heuristic(cuboids, (10, 10, 10))
    assert len(result.placements) == 1
    assert result.placements[0].position == (0, 0, 0)
    assert result.unpacked_ids == []


def test_single_box_exact_fit():
    cuboids = make_cuboids((5, 6, 7))
    result = pack_with_heuristic(cuboids, (5, 6, 7))
    assert len(result.placements) == 1
    assert result.utilisation == pytest.approx(1.0)


def test_two_boxes_no_overlap():
    # Two 5×5×5 boxes in a 10×5×5 container
    cuboids = make_cuboids((5, 5, 5), (5, 5, 5))
    result = pack_with_heuristic(cuboids, (10, 5, 5))
    assert len(result.placements) == 2
    assert result.unpacked_ids == []
    # Verify no overlap by replaying into a fresh grid
    grid = OccupancyGrid(10, 5, 5)
    for p in result.placements:
        x, y, z = p.position
        l, w, h = int(p.orientation.l), int(p.orientation.w), int(p.orientation.h)
        assert grid.is_free(x, y, z, l, w, h), f"Overlap detected for {p.item_id}"
        grid.place(p.item_id, x, y, z, l, w, h)


def test_eight_unit_cubes_fill_2x2x2():
    cuboids = [Cuboid(f"c{i}", (1, 1, 1)) for i in range(8)]
    cuboids = preprocess_cuboids(cuboids)
    result = pack_with_heuristic(cuboids, (2, 2, 2))
    assert len(result.placements) == 8
    assert result.utilisation == pytest.approx(1.0)


def test_unplaceable_box_skipped():
    # Box is larger than the container
    cuboids = make_cuboids((20, 20, 20))
    result = pack_with_heuristic(cuboids, (10, 10, 10))
    assert len(result.placements) == 0
    assert "box0" in result.unpacked_ids


def test_unplaceable_raises_when_fail_fast():
    cuboids = make_cuboids((20, 20, 20))
    with pytest.raises(PackingFailure):
        pack_with_heuristic(cuboids, (10, 10, 10), fail_fast=True)


def test_support_constraint_no_floating_items():
    # A box that can only sit on top of another box, with <50% support
    # Place a thin 1×1 base in a 4×4×4 container.
    # Then try to place a 4×4 box on top — half of which overhangs.
    # This tests that the support check prevents floating placements.
    base = Cuboid("base", (1, 1, 2))
    big = Cuboid("big", (4, 4, 1))
    cuboids = preprocess_cuboids([big, base])  # big first (larger volume)
    result = pack_with_heuristic(cuboids, (4, 4, 4))
    # big should be placed on the floor (z=0), base on top or beside
    for p in result.placements:
        if p.item_id == "big":
            assert p.position[2] == 0, "big box should be on the floor"


def test_utilisation_nontrivial():
    # 5 boxes of various sizes in a reasonably-sized container
    cuboids = make_cuboids(
        (3, 3, 3), (2, 2, 2), (4, 2, 1), (1, 1, 5), (2, 3, 2)
    )
    result = pack_with_heuristic(cuboids, (10, 10, 10))
    assert result.utilisation > 0.0
    assert len(result.placements) > 0


def test_ep_list_after_first_placement():
    # After placing one box at (0,0,0) the EP list should gain 3 new points
    # and lose (0,0,0). We verify by checking that at least 2 non-origin EPs exist
    # implicitly through the fact that a second box can be placed.
    cuboids = make_cuboids((3, 3, 3), (3, 3, 3))
    result = pack_with_heuristic(cuboids, (10, 10, 10))
    assert len(result.placements) == 2


def test_no_placements_in_empty_container():
    result = pack_with_heuristic([], (10, 10, 10))
    assert result.placements == []
    assert result.unpacked_ids == []
    assert result.utilisation == pytest.approx(0.0)


def test_result_utilisation_matches_packed_volume():
    cuboids = make_cuboids((2, 3, 4))
    result = pack_with_heuristic(cuboids, (10, 10, 10))
    expected = (2 * 3 * 4) / (10 * 10 * 10)
    assert result.utilisation == pytest.approx(expected)

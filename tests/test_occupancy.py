import pytest
from bin_packing.occupancy import OccupancyGrid


@pytest.fixture
def grid_10():
    return OccupancyGrid(10, 10, 10)


def test_empty_grid_all_free(grid_10):
    assert grid_10.is_free(0, 0, 0, 10, 10, 10)


def test_is_free_single_cell(grid_10):
    assert grid_10.is_free(5, 5, 5, 1, 1, 1)


def test_place_and_overlap(grid_10):
    grid_10.place("a", 0, 0, 0, 3, 3, 3)
    assert not grid_10.is_free(0, 0, 0, 3, 3, 3)
    assert not grid_10.is_free(1, 1, 1, 1, 1, 1)


def test_place_non_overlapping_region_still_free(grid_10):
    grid_10.place("a", 0, 0, 0, 3, 3, 3)
    assert grid_10.is_free(3, 0, 0, 7, 10, 10)


def test_out_of_bounds_not_free(grid_10):
    assert not grid_10.is_free(8, 8, 8, 5, 5, 5)  # exceeds 10 on all axes
    assert not grid_10.is_free(-1, 0, 0, 1, 1, 1)


def test_exact_boundary_region_is_free(grid_10):
    assert grid_10.is_free(8, 8, 8, 2, 2, 2)


def test_floor_support_is_full(grid_10):
    assert grid_10.supported_area_fraction(0, 0, 0, 5, 5) == pytest.approx(1.0)
    assert grid_10.supported_area_fraction(3, 3, 0, 2, 2) == pytest.approx(1.0)


def test_partial_support_above_placed_item(grid_10):
    # Place a 4×4 slab at (0,0,0); base is 4×4 = 16 cells.
    # Place a 4×4 box above it at z=3 — fully supported.
    grid_10.place("slab", 0, 0, 0, 4, 4, 3)
    frac = grid_10.supported_area_fraction(0, 0, 3, 4, 4)
    assert frac == pytest.approx(1.0)


def test_partial_support_half_supported(grid_10):
    # Slab covers x=[0,4), y=[0,4). Box sits at x=[2,6), y=[0,4) at z=2.
    # Half the base (2×4=8 out of 4×4=16) is supported.
    grid_10.place("slab", 0, 0, 0, 4, 4, 2)
    frac = grid_10.supported_area_fraction(2, 0, 2, 4, 4)
    assert frac == pytest.approx(0.5)


def test_zero_support_above_empty(grid_10):
    frac = grid_10.supported_area_fraction(0, 0, 5, 3, 3)
    assert frac == pytest.approx(0.0)


def test_copy_independence(grid_10):
    grid_10.place("a", 0, 0, 0, 2, 2, 2)
    assert not grid_10.is_free(0, 0, 0, 2, 2, 2)
    clone = grid_10.copy()
    clone.place("b", 2, 0, 0, 2, 2, 2)
    # Original should not see "b"
    assert grid_10.is_free(2, 0, 0, 2, 2, 2)
    # Clone should see both
    assert not clone.is_free(0, 0, 0, 2, 2, 2)
    assert not clone.is_free(2, 0, 0, 2, 2, 2)


def test_remove_item(grid_10):
    grid_10.place("a", 1, 1, 1, 3, 3, 3)
    assert not grid_10.is_free(1, 1, 1, 3, 3, 3)
    grid_10.remove("a")
    assert grid_10.is_free(1, 1, 1, 3, 3, 3)


def test_support_fraction_updates_after_remove(grid_10):
    grid_10.place("slab", 0, 0, 0, 4, 4, 2)
    assert grid_10.supported_area_fraction(0, 0, 2, 4, 4) == pytest.approx(1.0)
    grid_10.remove("slab")
    assert grid_10.supported_area_fraction(0, 0, 2, 4, 4) == pytest.approx(0.0)


def test_remove_nonexistent_is_safe(grid_10):
    grid_10.remove("does_not_exist")  # should not raise


def test_volume_accounting(grid_10):
    total = grid_10.L * grid_10.W * grid_10.H
    assert grid_10.free_volume() + grid_10.occupied_volume() == total
    grid_10.place("a", 0, 0, 0, 3, 3, 3)
    assert grid_10.free_volume() + grid_10.occupied_volume() == total
    assert grid_10.occupied_volume() == 27


def test_two_items_accounting(grid_10):
    grid_10.place("a", 0, 0, 0, 2, 2, 2)
    grid_10.place("b", 2, 0, 0, 2, 2, 2)
    assert grid_10.occupied_volume() == 8 + 8


def test_remove_restores_volume(grid_10):
    initial_free = grid_10.free_volume()
    grid_10.place("a", 0, 0, 0, 4, 4, 4)
    grid_10.remove("a")
    assert grid_10.free_volume() == initial_free

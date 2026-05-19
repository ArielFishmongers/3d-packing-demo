import pytest
from bin_packing.models import Cuboid
from bin_packing.orientations import unique_orientations, preprocess_cuboids


def test_cube_has_one_orientation():
    orients = unique_orientations((2.0, 2.0, 2.0))
    assert len(orients) == 1


def test_square_prism_has_three_orientations():
    orients = unique_orientations((2.0, 2.0, 4.0))
    assert len(orients) == 3


def test_generic_box_has_six_orientations():
    orients = unique_orientations((1.0, 2.0, 3.0))
    assert len(orients) == 6


def test_orientations_volume_preserved():
    dims = (2.0, 3.0, 5.0)
    expected_vol = 2.0 * 3.0 * 5.0
    for o in unique_orientations(dims):
        assert o.volume == pytest.approx(expected_vol)


def test_orientations_all_dims_present():
    dims = (1.0, 2.0, 3.0)
    for o in unique_orientations(dims):
        assert sorted(o.dims) == sorted(dims)


def test_preprocess_sorts_by_volume_descending():
    cuboids = [
        Cuboid("small", (1.0, 1.0, 1.0)),
        Cuboid("large", (3.0, 3.0, 3.0)),
        Cuboid("medium", (2.0, 2.0, 2.0)),
    ]
    sorted_cuboids = preprocess_cuboids(cuboids, sort_by="volume")
    vols = [c.volume for c in sorted_cuboids]
    assert vols == sorted(vols, reverse=True)
    assert sorted_cuboids[0].item_id == "large"


def test_preprocess_sorts_by_longest_edge():
    cuboids = [
        Cuboid("a", (1.0, 1.0, 5.0)),
        Cuboid("b", (3.0, 3.0, 3.0)),
        Cuboid("c", (2.0, 4.0, 2.0)),
    ]
    sorted_cuboids = preprocess_cuboids(cuboids, sort_by="longest_edge")
    assert sorted_cuboids[0].item_id == "a"
    assert sorted_cuboids[1].item_id == "c"
    assert sorted_cuboids[2].item_id == "b"


def test_preprocess_populates_orientations():
    cuboids = [Cuboid("box", (1.0, 2.0, 3.0))]
    result = preprocess_cuboids(cuboids)
    assert len(result[0].orientations) == 6


def test_preprocess_rotation_not_allowed():
    cuboids = [Cuboid("fixed", (1.0, 2.0, 3.0), rotations_allowed=False)]
    result = preprocess_cuboids(cuboids)
    assert len(result[0].orientations) == 1
    o = result[0].orientations[0]
    assert (o.l, o.w, o.h) == (1.0, 2.0, 3.0)


def test_preprocess_invalid_sort_raises():
    cuboids = [Cuboid("box", (1.0, 2.0, 3.0))]
    with pytest.raises(ValueError, match="Unknown sort_by"):
        preprocess_cuboids(cuboids, sort_by="invalid")

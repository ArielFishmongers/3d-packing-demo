import pytest
from bin_packing.models import Cuboid


@pytest.fixture
def small_container():
    return (10, 10, 10)


@pytest.fixture
def unit_cube():
    return Cuboid(item_id="cube1", dims=(1, 1, 1))


@pytest.fixture
def flat_box():
    return Cuboid(item_id="flat1", dims=(4, 4, 1))


@pytest.fixture
def five_random_boxes():
    import random
    rng = random.Random(42)
    return [
        Cuboid(
            item_id=f"box{i}",
            dims=(rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4)),
        )
        for i in range(5)
    ]

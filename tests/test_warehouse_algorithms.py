from __future__ import annotations

import pytest

from bin_packing.models import Cuboid, Orientation, PackingResult, Placement
from bin_packing.warehouse_algorithms import AlgorithmConfig, PACKING_MODES, pack_multi_container, pack_once


def make_result(item_id: str = "item") -> PackingResult:
    return PackingResult(
        placements=[
            Placement(
                item_id=item_id,
                orientation=Orientation(l=1, w=1, h=1),
                position=(0, 0, 0),
            )
        ],
        unpacked_ids=[],
        container_dims=(5, 5, 5),
    )


@pytest.mark.parametrize(
    ("mode", "expected_method", "expected_use_ga"),
    [
        ("sequential", "pack_sequential", False),
        ("heuristic", "pack", False),
        ("ga", "pack", True),
    ],
)
def test_pack_once_dispatches_to_expected_packer_method(
    monkeypatch,
    mode: str,
    expected_method: str,
    expected_use_ga: bool,
):
    calls: dict[str, object] = {}

    class FakePacker:
        def __init__(self, container_dims, config, seed=None):
            calls["container_dims"] = container_dims
            calls["use_ga"] = config.use_ga
            calls["sort_by"] = config.sort_by
            calls["precision"] = config.precision
            calls["seed"] = seed

        def pack(self, cuboids):
            calls["method"] = "pack"
            calls["cuboid_count"] = len(cuboids)
            return make_result("packed")

        def pack_sequential(self, cuboids):
            calls["method"] = "pack_sequential"
            calls["cuboid_count"] = len(cuboids)
            return make_result("sequential")

    monkeypatch.setattr("bin_packing.warehouse_algorithms.Packer", FakePacker)

    result = pack_once(
        [Cuboid("item", (1, 1, 1))],
        (5, 5, 5),
        AlgorithmConfig(mode=mode, precision=3, sort_by="longest_edge", seed=9),
    )

    assert calls["container_dims"] == (5, 5, 5)
    assert calls["use_ga"] is expected_use_ga
    assert calls["sort_by"] == "longest_edge"
    assert calls["precision"] == 3
    assert calls["seed"] == 9
    assert calls["method"] == expected_method
    assert calls["cuboid_count"] == 1
    assert result.placements


@pytest.mark.parametrize("mode", PACKING_MODES)
def test_pack_multi_container_retries_unpacked_items_in_fresh_containers(mode: str):
    cuboids = [Cuboid("a", (4, 4, 4)), Cuboid("b", (4, 4, 4))]

    results, warnings = pack_multi_container(
        cuboids,
        (4, 4, 4),
        algorithm=AlgorithmConfig(mode=mode, seed=0),
        quiet=True,
    )

    assert len(results) == 2
    assert [len(result.placements) for result in results] == [1, 1]
    assert warnings == []

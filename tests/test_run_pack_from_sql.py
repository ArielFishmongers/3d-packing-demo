from __future__ import annotations

from pathlib import Path

import pytest

from bin_packing import Cuboid
from bin_packing.io_postgres import QueryLoadResult
from bin_packing.models import Orientation, PackingResult, Placement
from scripts import run_pack_from_sql


def make_result() -> PackingResult:
    return PackingResult(
        placements=[
            Placement(
                item_id="item_1",
                orientation=Orientation(l=2, w=3, h=4),
                position=(0, 0, 0),
            )
        ],
        unpacked_ids=["item_2"],
        container_dims=(10, 10, 10),
    )


@pytest.mark.parametrize(
    ("mode", "expected_method", "expected_use_ga"),
    [
        ("heuristic", "pack", False),
        ("ga", "pack", True),
        ("sequential", "pack_sequential", False),
    ],
)
def test_main_dispatches_mode_and_prints_summary(
    tmp_path,
    monkeypatch,
    capsys,
    mode: str,
    expected_method: str,
    expected_use_ga: bool,
):
    query_file = tmp_path / "items.sql"
    query_file.write_text("SELECT 1;", encoding="utf-8")

    load_result = QueryLoadResult(
        row_count=2,
        cuboids=[Cuboid("item_1", (2, 3, 4)), Cuboid("item_2", (20, 20, 20))],
    )
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setattr(run_pack_from_sql, "load_cuboids_from_sql_file", lambda *_: load_result)

    calls: dict[str, object] = {}
    result = make_result()

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
            return result

        def pack_sequential(self, cuboids):
            calls["method"] = "pack_sequential"
            calls["cuboid_count"] = len(cuboids)
            return result

    monkeypatch.setattr(run_pack_from_sql, "Packer", FakePacker)

    exit_code = run_pack_from_sql.main(["--query-file", str(query_file), "--mode", mode])

    assert exit_code == 0
    assert calls["container_dims"] == run_pack_from_sql.DEFAULT_CONTAINER_DIMS
    assert calls["use_ga"] is expected_use_ga
    assert calls["method"] == expected_method
    assert calls["cuboid_count"] == 2

    captured = capsys.readouterr()
    assert "Fetched rows: 2" in captured.out
    assert "Expanded items: 2" in captured.out
    assert "Packed items: 1" in captured.out
    assert "Unpacked IDs:" in captured.out
    assert "item_2" in captured.out


def test_main_returns_error_when_loader_fails(tmp_path, monkeypatch, capsys):
    query_file = tmp_path / "items.sql"
    query_file.write_text("SELECT 1;", encoding="utf-8")

    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    monkeypatch.setattr(
        run_pack_from_sql,
        "load_cuboids_from_sql_file",
        lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    exit_code = run_pack_from_sql.main(["--query-file", str(query_file)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: boom" in captured.err

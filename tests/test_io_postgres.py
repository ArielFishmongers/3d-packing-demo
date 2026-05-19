from __future__ import annotations

import pytest

from bin_packing.io_postgres import QueryLoadResult, load_cuboids_from_sql_file, rows_to_cuboids


def test_rows_to_cuboids_defaults_optional_fields():
    cuboids = rows_to_cuboids(
        [{"item_id": "box", "length": 4, "width": 5, "height": 6}]
    )

    assert len(cuboids) == 1
    assert cuboids[0].item_id == "box"
    assert cuboids[0].dims == (4, 5, 6)
    assert cuboids[0].rotations_allowed is True
    assert cuboids[0].weight == pytest.approx(1.0)


def test_rows_to_cuboids_expands_quantity_with_suffixes():
    cuboids = rows_to_cuboids(
        [{"item_id": "crate", "length": 2, "width": 3, "height": 4, "quantity": 3}]
    )

    assert [cuboid.item_id for cuboid in cuboids] == ["crate_1", "crate_2", "crate_3"]


def test_rows_to_cuboids_duplicate_item_ids_across_rows_get_unique_suffixes():
    cuboids = rows_to_cuboids(
        [
            {"item_id": "sku", "length": 1, "width": 1, "height": 1},
            {"item_id": "sku", "length": 2, "width": 2, "height": 2},
        ]
    )

    assert [cuboid.item_id for cuboid in cuboids] == ["sku_1", "sku_2"]


def test_rows_to_cuboids_allows_boolean_like_fields():
    cuboids = rows_to_cuboids(
        [
            {
                "item_id": "fixed",
                "length": 1,
                "width": 2,
                "height": 3,
                "rotations_allowed": 0,
                "weight": "2.5",
            }
        ]
    )

    assert cuboids[0].rotations_allowed is False
    assert cuboids[0].weight == pytest.approx(2.5)


def test_rows_to_cuboids_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="missing required column"):
        rows_to_cuboids([{"item_id": "bad", "length": 1, "width": 2}])


def test_rows_to_cuboids_rejects_null_dimensions():
    with pytest.raises(ValueError, match="cannot be null"):
        rows_to_cuboids([{"item_id": "bad", "length": None, "width": 2, "height": 3}])


def test_rows_to_cuboids_rejects_non_positive_dimensions():
    with pytest.raises(ValueError, match="must be positive"):
        rows_to_cuboids([{"item_id": "bad", "length": 0, "width": 2, "height": 3}])


def test_rows_to_cuboids_rejects_fractional_dimensions():
    with pytest.raises(ValueError, match="must be an integer value"):
        rows_to_cuboids([{"item_id": "bad", "length": 1.5, "width": 2, "height": 3}])


def test_load_cuboids_from_sql_file_reads_query_and_maps_rows(tmp_path, monkeypatch):
    query_file = tmp_path / "items.sql"
    query_file.write_text("SELECT 1;", encoding="utf-8")

    def fake_execute_query(database_url: str, query: str):
        assert database_url == "postgresql://example"
        assert query == "SELECT 1;"
        return [{"item_id": "row1", "length": 1, "width": 2, "height": 3, "quantity": 2}]

    monkeypatch.setattr("bin_packing.io_postgres.execute_query", fake_execute_query)

    load_result = load_cuboids_from_sql_file("postgresql://example", query_file)

    assert isinstance(load_result, QueryLoadResult)
    assert load_result.row_count == 1
    assert [cuboid.item_id for cuboid in load_result.cuboids] == ["row1_1", "row1_2"]

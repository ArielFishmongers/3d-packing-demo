from __future__ import annotations

import csv

from bin_packing.warehouse_io import (
    WarehouseDataPaths,
    load_events_by_job,
    load_item_master,
    load_picking_events,
    parse_container_dims,
)


def write_csv(path, fieldnames, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_paths(tmp_path) -> WarehouseDataPaths:
    activity_file = tmp_path / "activity.csv"
    item_file = tmp_path / "items.csv"
    write_csv(
        item_file,
        ["consignee", "sku", "length", "width", "height"],
        [
            {"consignee": "C1", "sku": "SKU1", "length": "1", "width": "2", "height": "3"},
            {"consignee": "C1", "sku": "BAD", "length": "x", "width": "2", "height": "3"},
            {"consignee": "C2", "sku": "SKU2", "length": "2.5", "width": "1", "height": "1"},
        ],
    )
    write_csv(
        activity_file,
        [
            "event_process_type",
            "event_type",
            "container",
            "job",
            "event_end",
            "quantity",
            "consignee",
            "sku",
        ],
        [
            {
                "event_process_type": "PACKING",
                "event_type": "PUT",
                "container": "C-IGNORED",
                "job": "job-ignored",
                "event_end": "2026-03-13T10:02:00",
                "quantity": "1",
                "consignee": "C1",
                "sku": "SKU1",
            },
            {
                "event_process_type": "PICKING",
                "event_type": "SCAN",
                "container": "C-IGNORED",
                "job": "job-ignored",
                "event_end": "2026-03-13T10:03:00",
                "quantity": "1",
                "consignee": "C1",
                "sku": "SKU1",
            },
            {
                "event_process_type": "PICKING",
                "event_type": "PUT",
                "container": "00000000",
                "job": "job-ignored",
                "event_end": "2026-03-13T10:04:00",
                "quantity": "1",
                "consignee": "C1",
                "sku": "SKU1",
            },
            {
                "event_process_type": "PICKING",
                "event_type": "PUT",
                "container": "C2",
                "job": "job-b",
                "event_end": "2026-03-13T10:01:00",
                "quantity": "2",
                "consignee": "C2",
                "sku": "SKU2",
            },
            {
                "event_process_type": "PICKING",
                "event_type": "PUT",
                "container": "C1",
                "job": "job-a",
                "event_end": "2026-03-13T10:00:00",
                "quantity": "1",
                "consignee": "C1",
                "sku": "SKU1",
            },
        ],
    )
    return WarehouseDataPaths(activity_file=activity_file, item_file=item_file)


def test_parse_container_dims_accepts_x_and_spaces():
    assert parse_container_dims("27x32x50") == (27, 32, 50)
    assert parse_container_dims("27 32 50") == (27, 32, 50)
    assert parse_container_dims("bad") is None


def test_load_item_master_skips_invalid_rows(tmp_path):
    paths = make_paths(tmp_path)

    items = load_item_master(paths)

    assert items[("C1", "SKU1")]["dims"] == (1.0, 2.0, 3.0)
    assert items[("C1", "SKU1")]["volume"] == 6.0
    assert items[("C2", "SKU2")]["dims"] == (2.5, 1.0, 1.0)
    assert ("C1", "BAD") not in items


def test_load_picking_events_filters_orders_and_limits(tmp_path):
    paths = make_paths(tmp_path)

    events = load_picking_events(paths=paths)

    assert [event["job"] for event in events] == ["job-a", "job-b"]
    assert [event["event_end"] for event in events] == [
        "2026-03-13T10:00:00",
        "2026-03-13T10:01:00",
    ]

    limited = load_picking_events(limit=1, paths=paths)
    assert len(limited) == 1
    assert limited[0]["job"] == "job-a"

    filtered = load_picking_events(job="job-b", paths=paths)
    assert len(filtered) == 1
    assert filtered[0]["container"] == "C2"


def test_load_events_by_job_groups_sorted_events(tmp_path):
    paths = make_paths(tmp_path)

    grouped = load_events_by_job(paths=paths)

    assert list(grouped) == ["job-a", "job-b"]
    assert grouped["job-a"][0]["event_end"] == "2026-03-13T10:00:00"
    assert grouped["job-b"][0]["event_end"] == "2026-03-13T10:01:00"

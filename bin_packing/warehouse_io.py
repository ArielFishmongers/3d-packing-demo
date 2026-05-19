from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ACTIVITY_FILE = DEFAULT_DATA_DIR / "sample-warehouse-activity-feed.csv"
DEFAULT_ITEM_FILE = DEFAULT_DATA_DIR / "sample-warehouse-item-master.csv"


@dataclass(frozen=True)
class WarehouseDataPaths:
    activity_file: Path = DEFAULT_ACTIVITY_FILE
    item_file: Path = DEFAULT_ITEM_FILE


DEFAULT_WAREHOUSE_DATA_PATHS = WarehouseDataPaths()


def parse_container_dims(spec: str) -> tuple[int, int, int] | None:
    """Parse 'LxWxH' or 'L W H' into (L, W, H) ints."""
    match = re.search(r"(\d+)[x\s](\d+)[x\s](\d+)", spec)
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def load_item_master(
    paths: WarehouseDataPaths = DEFAULT_WAREHOUSE_DATA_PATHS,
) -> dict[tuple[str, str], dict[str, float | tuple[float, float, float]]]:
    """Return {(consignee, sku): {'dims': (l, w, h), 'volume': float}}."""
    items: dict[tuple[str, str], dict[str, float | tuple[float, float, float]]] = {}
    with paths.item_file.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                length = float(row["length"])
                width = float(row["width"])
                height = float(row["height"])
                items[(row["consignee"], row["sku"])] = {
                    "dims": (length, width, height),
                    "volume": length * width * height,
                }
            except (KeyError, TypeError, ValueError):
                continue
    return items


def load_picking_events(
    job: str | None = None,
    limit: int | None = None,
    paths: WarehouseDataPaths = DEFAULT_WAREHOUSE_DATA_PATHS,
) -> list[dict[str, str]]:
    """Return PICKING/PUT rows sorted by event_end."""
    events: list[dict[str, str]] = []
    with paths.activity_file.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (
                row["event_process_type"] == "PICKING"
                and row["event_type"] == "PUT"
                and row["container"] != "00000000"
                and (job is None or row["job"] == job)
            ):
                events.append(row)

    events.sort(key=lambda row: row["event_end"])
    if limit is not None:
        events = events[:limit]
    return events


def load_events_by_job(
    limit: int | None = None,
    paths: WarehouseDataPaths = DEFAULT_WAREHOUSE_DATA_PATHS,
) -> dict[str, list[dict[str, str]]]:
    """Return all events grouped by job ID, each group sorted by event_end."""
    by_job: dict[str, list[dict[str, str]]] = {}
    for event in load_picking_events(limit=limit, paths=paths):
        by_job.setdefault(event["job"], []).append(event)
    return by_job

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

from .models import Cuboid, PackingResult
from .warehouse_algorithms import AlgorithmConfig, pack_multi_container

ItemDims = dict[tuple[str, str], dict[str, float | tuple[float, float, float]]]


@dataclass(frozen=True)
class ComparisonConfig:
    container_dims: tuple[int, int, int]
    max_workers: int | None = None
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)

    def __post_init__(self) -> None:
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")


def _row_quantity(row: dict[str, str]) -> int:
    return int(row["quantity"]) if row.get("quantity") else 1


def build_cuboids(events: list[dict[str, str]], item_dims: ItemDims) -> tuple[list[Cuboid], int]:
    """Expand pick lines into individual Cuboid units (quantity=2 -> two cuboids)."""
    cuboids: list[Cuboid] = []
    skipped = 0
    for index, row in enumerate(events):
        key = (row["consignee"], row["sku"])
        if key not in item_dims:
            skipped += 1
            continue
        dims = item_dims[key]["dims"]
        assert isinstance(dims, tuple)
        for unit in range(_row_quantity(row)):
            cuboids.append(Cuboid(item_id=f"{row['sku']}_{index}_{unit}", dims=dims))
    return cuboids, skipped


def worker_stats(
    events: list[dict[str, str]],
    item_dims: ItemDims,
    container_dims: tuple[int, int, int],
) -> dict[str, float | int | None]:
    containers: dict[str, dict[str, float | int]] = {}
    for row in events:
        container = containers.setdefault(
            row["container"],
            {"lines": 0, "items": 0, "volume": 0.0, "missing": 0},
        )
        container["lines"] += 1
        qty = _row_quantity(row)
        container["items"] += qty
        key = (row["consignee"], row["sku"])
        if key in item_dims:
            volume = item_dims[key]["volume"]
            assert isinstance(volume, float)
            container["volume"] += volume * qty
        else:
            container["missing"] += 1

    container_volume = container_dims[0] * container_dims[1] * container_dims[2]
    utils: list[float] = []
    items_per: list[int] = []
    lines_per: list[int] = []
    for container in containers.values():
        items_per.append(int(container["items"]))
        lines_per.append(int(container["lines"]))
        volume = float(container["volume"])
        if container_volume and volume > 0:
            utils.append(volume / container_volume * 100)

    return {
        "total_containers": len(containers),
        "total_items": sum(int(container["items"]) for container in containers.values()),
        "total_lines": sum(int(container["lines"]) for container in containers.values()),
        "avg_items": statistics.mean(items_per) if items_per else 0,
        "min_items": min(items_per) if items_per else 0,
        "max_items": max(items_per) if items_per else 0,
        "avg_lines": statistics.mean(lines_per) if lines_per else 0,
        "avg_util": statistics.mean(utils) if utils else 0,
        "min_util": min(utils) if utils else 0,
        "max_util": max(utils) if utils else 0,
        "missing_dims": sum(int(container["missing"]) for container in containers.values()),
    }


def build_warehouse_container_list(
    events: list[dict[str, str]],
    item_dims: ItemDims,
    container_dims: tuple[int, int, int],
) -> list[dict[str, float | int | str]]:
    """Return containers in picking order (by first event_end) with fill stats."""
    order: list[str] = []
    containers: dict[str, dict[str, float | int]] = {}
    for row in events:
        container_id = row["container"]
        if container_id not in containers:
            order.append(container_id)
        container = containers.setdefault(container_id, {"items": 0, "volume": 0.0})
        qty = _row_quantity(row)
        container["items"] += qty
        key = (row["consignee"], row["sku"])
        if key in item_dims:
            volume = item_dims[key]["volume"]
            assert isinstance(volume, float)
            container["volume"] += volume * qty

    container_volume = container_dims[0] * container_dims[1] * container_dims[2]
    result: list[dict[str, float | int | str]] = []
    for rank, container_id in enumerate(order, start=1):
        container = containers[container_id]
        volume = float(container["volume"])
        util_pct = volume / container_volume * 100 if container_volume else 0.0
        result.append(
            {
                "rank": rank,
                "container": container_id,
                "items": int(container["items"]),
                "util_pct": util_pct,
            }
        )
    return result


def algorithm_stats(
    results: list[PackingResult],
    container_dims: tuple[int, int, int],
) -> tuple[dict[str, float | int | None], list[dict[str, float | int]]]:
    """Return (aggregate_stats, per_container_list) for algorithm results."""
    container_volume = container_dims[0] * container_dims[1] * container_dims[2]
    items_per: list[int] = []
    utils: list[float] = []
    container_list: list[dict[str, float | int]] = []

    for rank, result in enumerate(results, start=1):
        item_count = len(result.placements)
        packed_volume = sum(
            placement.orientation.l * placement.orientation.w * placement.orientation.h
            for placement in result.placements
        )
        util_pct = packed_volume / container_volume * 100 if container_volume else 0.0
        items_per.append(item_count)
        utils.append(util_pct)
        container_list.append({"rank": rank, "items": item_count, "util_pct": util_pct})

    aggregate = {
        "total_containers": len(results),
        "total_items": sum(items_per),
        "total_lines": None,
        "avg_items": statistics.mean(items_per) if items_per else 0,
        "min_items": min(items_per) if items_per else 0,
        "max_items": max(items_per) if items_per else 0,
        "avg_lines": None,
        "avg_util": statistics.mean(utils) if utils else 0,
        "min_util": min(utils) if utils else 0,
        "max_util": max(utils) if utils else 0,
        "missing_dims": 0,
    }
    return aggregate, container_list


def compare_job(
    job_id: str,
    events: list[dict[str, str]],
    item_dims: ItemDims,
    config: ComparisonConfig,
) -> dict[str, object]:
    cuboids, skipped = build_cuboids(events, item_dims)
    results, skip_warnings = pack_multi_container(
        cuboids,
        config.container_dims,
        algorithm=config.algorithm,
        quiet=True,
    )
    ws = worker_stats(events, item_dims, config.container_dims)
    w_containers = build_warehouse_container_list(events, item_dims, config.container_dims)
    als, a_containers = algorithm_stats(results, config.container_dims)

    return {
        "job_id": job_id,
        "ws": ws,
        "als": als,
        "w_containers": w_containers,
        "a_containers": a_containers,
        "pack_results": results,
        "id_to_dims": {c.item_id: c.dims for c in cuboids},
        "skipped": skipped,
        "n_cuboids": len(cuboids),
        "skip_warnings": skip_warnings,
    }


def _process_job(args: tuple[str, list[dict[str, str]], ItemDims, ComparisonConfig]) -> dict[str, object]:
    """Module-level worker function for ProcessPoolExecutor."""
    job_id, events, item_dims, config = args
    return compare_job(job_id, events, item_dims, config)


def run_jobs_parallel(
    events_by_job: dict[str, list[dict[str, str]]],
    item_dims: ItemDims,
    config: ComparisonConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, object]]:
    """Pack every job independently and return results in job-start order.

    Args:
        progress_callback: Optional callable(completed, total) called after
                           each job finishes. Useful for live progress bars.
    """
    ordered_jobs = sorted(events_by_job.items(), key=lambda item: item[1][0]["event_end"])
    if not ordered_jobs:
        return []

    args_list = [(job_id, events, item_dims, config) for job_id, events in ordered_jobs]
    total = len(args_list)
    max_workers = config.max_workers
    if max_workers is None:
        max_workers = max(1, os.cpu_count() or 1)

    if max_workers == 1:
        results = []
        for i, args in enumerate(args_list, start=1):
            results.append(_process_job(args))
            if progress_callback:
                progress_callback(i, total)
        return results

    completed = 0
    job_results: dict[str, dict[str, object]] = {}
    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(_process_job, args): args[0] for args in args_list}
        for future in as_completed(future_to_job):
            completed += 1
            job_id = future_to_job[future]
            elapsed = time.perf_counter() - start
            print(
                f"  [{completed:>{len(str(total))}}/{total}] job {job_id}  "
                f"({elapsed:.1f}s elapsed)",
                end="\r",
                flush=True,
            )
            result = future.result()
            job_results[result["job_id"]] = result
            if progress_callback:
                progress_callback(completed, total)

    elapsed = time.perf_counter() - start
    print(f"  Done: {total} job(s) in {elapsed:.2f}s{' ' * 30}")
    return [job_results[job_id] for job_id, _ in ordered_jobs]


def _aggregate_stats(stats_list: list[dict[str, float | int | None]]) -> dict[str, float | int | None]:
    if not stats_list:
        return {
            "total_containers": 0,
            "total_items": 0,
            "total_lines": None,
            "avg_items": 0,
            "min_items": 0,
            "max_items": 0,
            "avg_lines": None,
            "avg_util": 0,
            "min_util": 0,
            "max_util": 0,
            "missing_dims": 0,
        }

    total_containers = sum(int(stats["total_containers"]) for stats in stats_list)
    total_items = sum(int(stats["total_items"]) for stats in stats_list)
    total_lines_values = [int(stats["total_lines"]) for stats in stats_list if stats["total_lines"]]
    util_values = [float(stats["avg_util"]) for stats in stats_list if float(stats["avg_util"]) > 0]
    item_values = [float(stats["avg_items"]) for stats in stats_list]

    return {
        "total_containers": total_containers,
        "total_items": total_items,
        "total_lines": sum(total_lines_values) or None,
        "avg_items": statistics.mean(item_values) if item_values else 0,
        "min_items": min(int(stats["min_items"]) for stats in stats_list),
        "max_items": max(int(stats["max_items"]) for stats in stats_list),
        "avg_lines": None,
        "avg_util": statistics.mean(util_values) if util_values else 0,
        "min_util": min(float(stats["min_util"]) for stats in stats_list),
        "max_util": max(float(stats["max_util"]) for stats in stats_list),
        "missing_dims": sum(int(stats["missing_dims"]) for stats in stats_list),
    }


def aggregate_across_jobs(
    job_results: list[dict[str, object]],
) -> tuple[dict[str, float | int | None], dict[str, float | int | None]]:
    """Sum and average per-job stats into a single aggregate dict pair."""
    ws_agg = _aggregate_stats([result["ws"] for result in job_results])
    als_agg = _aggregate_stats([result["als"] for result in job_results])
    return ws_agg, als_agg

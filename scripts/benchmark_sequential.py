from __future__ import annotations

import time
from dataclasses import dataclass

from bin_packing.warehouse_algorithms import AlgorithmConfig
from bin_packing.warehouse_compare import ComparisonConfig, compare_job
from bin_packing.warehouse_io import load_events_by_job, load_item_master, parse_container_dims


@dataclass(frozen=True)
class Scenario:
    name: str
    job_id: str
    line_limit: int | None


def _time_job(
    job_id: str,
    events: list[dict[str, str]],
    item_dims: dict,
    container_dims: tuple[int, int, int],
    mode: str,
) -> tuple[float, dict[str, object]]:
    config = ComparisonConfig(
        container_dims=container_dims,
        max_workers=1,
        algorithm=AlgorithmConfig(mode=mode),
    )
    started = time.perf_counter()
    result = compare_job(job_id, events, item_dims, config)
    elapsed = time.perf_counter() - started
    return elapsed, result


def main() -> int:
    container_dims = parse_container_dims("27x32x50")
    assert container_dims is not None

    item_dims = load_item_master()
    events_by_job = load_events_by_job()
    ordered_jobs = sorted(events_by_job.items(), key=lambda item: len(item[1]))
    median_job_id = ordered_jobs[len(ordered_jobs) // 2][0]
    largest_job_id = ordered_jobs[-1][0]
    sample_job_id = "01728848-42" if "01728848-42" in events_by_job else median_job_id

    scenarios = [
        Scenario("Sample job (10 pick lines)", sample_job_id, 10),
        Scenario("Median job subset (20 pick lines)", median_job_id, 20),
        Scenario("Median job subset (40 pick lines)", median_job_id, 40),
        Scenario("Largest job (full)", largest_job_id, None),
    ]

    print(f"Container: {container_dims[0]}x{container_dims[1]}x{container_dims[2]}")
    print(f"Jobs available: {len(events_by_job)}")
    print()

    for scenario in scenarios:
        all_events = events_by_job[scenario.job_id]
        events = all_events[: scenario.line_limit] if scenario.line_limit is not None else all_events
        print(f"{scenario.name}")
        print(f"  job_id: {scenario.job_id}")
        print(f"  pick lines: {len(events)}")

        sequential_time, sequential_result = _time_job(
            scenario.job_id,
            events,
            item_dims,
            container_dims,
            "sequential",
        )
        heuristic_time, heuristic_result = _time_job(
            scenario.job_id,
            events,
            item_dims,
            container_dims,
            "heuristic",
        )

        print(
            "  sequential:"
            f" {sequential_time:.3f}s,"
            f" cuboids={sequential_result['n_cuboids']},"
            f" containers={sequential_result['als']['total_containers']}"
        )
        print(
            "  heuristic: "
            f"{heuristic_time:.3f}s,"
            f" cuboids={heuristic_result['n_cuboids']},"
            f" containers={heuristic_result['als']['total_containers']}"
        )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

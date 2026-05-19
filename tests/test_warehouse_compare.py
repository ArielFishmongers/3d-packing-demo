from __future__ import annotations

from bin_packing.warehouse_algorithms import AlgorithmConfig
from bin_packing.warehouse_compare import (
    ComparisonConfig,
    aggregate_across_jobs,
    build_cuboids,
    compare_job,
    run_jobs_parallel,
    worker_stats,
)


def make_item_dims():
    return {
        ("C1", "SKU1"): {"dims": (1.0, 1.0, 1.0), "volume": 1.0},
        ("C1", "SKU2"): {"dims": (2.0, 1.0, 1.0), "volume": 2.0},
    }


def make_event(job: str, container: str, event_end: str, sku: str, quantity: str = "1"):
    return {
        "job": job,
        "container": container,
        "event_end": event_end,
        "quantity": quantity,
        "consignee": "C1",
        "sku": sku,
    }


def test_build_cuboids_expands_quantity_and_counts_missing_dims():
    cuboids, skipped = build_cuboids(
        [
            make_event("job-1", "W1", "2026-03-13T10:00:00", "SKU1", quantity="2"),
            make_event("job-1", "W1", "2026-03-13T10:01:00", "MISSING"),
        ],
        make_item_dims(),
    )

    assert skipped == 1
    assert [cuboid.item_id for cuboid in cuboids] == ["SKU1_0_0", "SKU1_0_1"]


def test_worker_stats_counts_missing_dimension_lines():
    stats = worker_stats(
        [
            make_event("job-1", "W1", "2026-03-13T10:00:00", "SKU1", quantity="2"),
            make_event("job-1", "W1", "2026-03-13T10:01:00", "MISSING"),
        ],
        make_item_dims(),
        (4, 4, 4),
    )

    assert stats["total_containers"] == 1
    assert stats["total_items"] == 3
    assert stats["missing_dims"] == 1


def test_compare_job_returns_dashboard_contract():
    config = ComparisonConfig(
        container_dims=(4, 4, 4),
        max_workers=1,
        algorithm=AlgorithmConfig(mode="sequential"),
    )
    result = compare_job(
        "job-1",
        [
            make_event("job-1", "W1", "2026-03-13T10:00:00", "SKU1", quantity="2"),
            make_event("job-1", "W1", "2026-03-13T10:01:00", "MISSING"),
        ],
        make_item_dims(),
        config,
    )

    assert set(result) == {
        "job_id",
        "ws",
        "als",
        "w_containers",
        "a_containers",
        "pack_results",
        "id_to_dims",
        "skipped",
        "n_cuboids",
        "skip_warnings",
    }
    assert result["job_id"] == "job-1"
    assert result["skipped"] == 1
    assert result["n_cuboids"] == 2
    assert isinstance(result["w_containers"], list)
    assert isinstance(result["a_containers"], list)


def test_run_jobs_parallel_preserves_job_order_and_aggregates():
    config = ComparisonConfig(
        container_dims=(4, 4, 4),
        max_workers=1,
        algorithm=AlgorithmConfig(mode="sequential"),
    )
    events_by_job = {
        "job-late": [make_event("job-late", "W2", "2026-03-13T10:05:00", "SKU2")],
        "job-early": [make_event("job-early", "W1", "2026-03-13T10:00:00", "SKU1")],
    }

    job_results = run_jobs_parallel(events_by_job, make_item_dims(), config)
    ws_agg, als_agg = aggregate_across_jobs(job_results)

    assert [result["job_id"] for result in job_results] == ["job-early", "job-late"]
    assert ws_agg["total_containers"] == 2
    assert ws_agg["total_items"] == 2
    assert als_agg["total_items"] == 2
    assert "avg_util" in ws_agg
    assert "avg_util" in als_agg

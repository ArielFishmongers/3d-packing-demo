"""
compare_warehouse.py — Compare actual worker picking vs the packing algorithms.

Run with:
    python examples/compare_warehouse.py                     # all jobs (parallel)
    python examples/compare_warehouse.py --job 01114371-214 # single job
    python examples/compare_warehouse.py --container 27x32x50
    python examples/compare_warehouse.py --mode heuristic
    python examples/compare_warehouse.py --limit 500        # cap pick lines for testing
    python examples/compare_warehouse.py --workers 8        # parallel worker processes
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import defaultdict
from typing import Sequence

from bin_packing.warehouse_algorithms import AlgorithmConfig, PACKING_MODES
from bin_packing.warehouse_compare import (
    ComparisonConfig,
    aggregate_across_jobs,
    compare_job,
    run_jobs_parallel,
)
from bin_packing.warehouse_io import (
    load_events_by_job,
    load_item_master,
    load_picking_events,
    parse_container_dims,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare warehouse worker packing vs the packing algorithms."
    )
    parser.add_argument(
        "--job",
        metavar="JOB_ID",
        default=None,
        help=(
            "Filter to a single job ID (e.g. --job 01114371-214). "
            "Omit to process all jobs in parallel. "
            "Run with --list-jobs to see available IDs."
        ),
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="Print all available job IDs with their stats and exit.",
    )
    parser.add_argument(
        "--container",
        metavar="LxWxH",
        default="27x32x50",
        help="Container dimensions in cm, e.g. 27x32x50 (default: 27x32x50).",
    )
    parser.add_argument(
        "--mode",
        choices=PACKING_MODES,
        default="sequential",
        help="Packing algorithm to run (default: sequential).",
    )
    parser.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=None,
        help="Cap total input to the first N pick lines (sorted by event_end).",
    )
    parser.add_argument(
        "--workers",
        metavar="N",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: CPU count).",
    )
    return parser


def fmt(val, suffix="") -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.1f}{suffix}"
    return f"{val}{suffix}"


def print_comparison(ws: dict, als: dict, container_dims: tuple[int, int, int]) -> None:
    L, W, H = container_dims
    cvol = L * W * H

    label_width = 38
    col_width = 13
    sep = f"  {'─' * label_width}  {'─' * col_width}  {'─' * col_width}"
    head = f"  {'Metric':<{label_width}}  {'Workers':>{col_width}}  {'Algorithm':>{col_width}}"

    def row(label, w_val, a_val):
        print(f"  {label:<{label_width}}  {w_val:>{col_width}}  {a_val:>{col_width}}")

    print()
    print(f"  Container: {L}×{W}×{H} cm  ({cvol:,} cm³)")
    print(sep)
    print(head)
    print(sep)
    row("Total containers used", fmt(ws["total_containers"]), fmt(als["total_containers"]))
    row("Total items packed", fmt(ws["total_items"]), fmt(als["total_items"]))
    row("Total pick lines", fmt(ws["total_lines"]), fmt(als["total_lines"]))
    print(sep)
    row("Avg items / container", fmt(ws["avg_items"]), fmt(als["avg_items"]))
    row("Min items in a container", fmt(ws["min_items"]), fmt(als["min_items"]))
    row("Max items in a container", fmt(ws["max_items"]), fmt(als["max_items"]))
    row("Avg pick lines / container", fmt(ws["avg_lines"]), fmt(als["avg_lines"]))
    print(sep)
    row("Avg volume utilisation", fmt(ws["avg_util"], "%"), fmt(als["avg_util"], "%"))
    row("Min volume utilisation", fmt(ws["min_util"], "%"), fmt(als["min_util"], "%"))
    row("Max volume utilisation", fmt(ws["max_util"], "%"), fmt(als["max_util"], "%"))
    print(sep)
    row("Lines with missing dims", fmt(ws["missing_dims"]), fmt(als["missing_dims"]))
    print(sep)

    saved = ws["total_containers"] - als["total_containers"]
    util_delta = als["avg_util"] - ws["avg_util"]
    print()
    if saved > 0:
        pct = saved / ws["total_containers"] * 100
        print(f"  → Algorithm uses {saved} fewer containers ({pct:.0f}% reduction)")
    elif saved < 0:
        print(f"  → Algorithm uses {-saved} more containers than workers")
    if util_delta > 0:
        print(f"  → Volume utilisation improved by {util_delta:.1f} percentage points")
    print()


def print_per_job_comparison(job_results: list[dict[str, object]]) -> None:
    """Per-job summary: worker containers vs algorithm containers."""
    job_width = 24
    count_width = 8
    fill_width = 9

    sep = (
        f"  {'─' * job_width}  {'─' * count_width}  {'─' * fill_width}"
        f"     {'─' * count_width}  {'─' * fill_width}  {'─' * 5}"
    )
    header = (
        f"  {'Job ID':<{job_width}}  {'W-ctrs':>{count_width}}  {'W-fill%':>{fill_width}}"
        f"     {'A-ctrs':>{count_width}}  {'A-fill%':>{fill_width}}  {'Saved':>{5}}"
    )

    print("  Per-job breakdown  (one row per job, ordered by first pick time)")
    print(sep)
    print(header)
    print(sep)

    for result in job_results:
        ws = result["ws"]
        als = result["als"]
        assert isinstance(ws, dict)
        assert isinstance(als, dict)
        saved = ws["total_containers"] - als["total_containers"]
        saved_str = f"+{saved}" if saved > 0 else str(saved)

        print(
            f"  {result['job_id']:<{job_width}}"
            f"  {fmt(ws['total_containers']):>{count_width}}"
            f"  {fmt(ws['avg_util'], '%'):>{fill_width}}"
            f"     {fmt(als['total_containers']):>{count_width}}"
            f"  {fmt(als['avg_util'], '%'):>{fill_width}}"
            f"  {saved_str:>{5}}"
        )

    print(sep)


def print_per_container_comparison(
    w_containers: list[dict[str, object]],
    a_containers: list[dict[str, object]],
    container_dims: tuple[int, int, int],
) -> None:
    """Side-by-side: warehouse container N (by pick time) vs algorithm container N."""
    L, W, H = container_dims

    rank_width = 4
    id_width = 12
    item_width = 7
    fill_width = 9

    sep = (
        f"  {'─' * rank_width}  {'─' * id_width}  {'─' * item_width}  {'─' * fill_width}"
        f"     {'─' * item_width}  {'─' * fill_width}"
    )
    header = (
        f"  {'#':>{rank_width}}  {'Container':>{id_width}}"
        f"  {'W-items':>{item_width}}  {'W-fill%':>{fill_width}}"
        f"     {'A-items':>{item_width}}  {'A-fill%':>{fill_width}}"
    )

    n_rows = max(len(w_containers), len(a_containers))

    print("  Per-container breakdown  (warehouse order vs algorithm order)")
    print(f"  Container: {L}×{W}×{H} cm — ordered by first pick time")
    print(sep)
    print(header)
    print(sep)

    for index in range(n_rows):
        w_container = w_containers[index] if index < len(w_containers) else None
        a_container = a_containers[index] if index < len(a_containers) else None

        rank = (w_container or a_container)["rank"]
        warehouse_id = w_container["container"] if w_container else "—"
        warehouse_items = fmt(w_container["items"]) if w_container else "—"
        warehouse_fill = fmt(w_container["util_pct"], "%") if w_container else "—"
        algorithm_items = fmt(a_container["items"]) if a_container else "—"
        algorithm_fill = fmt(a_container["util_pct"], "%") if a_container else "—"

        print(
            f"  {rank:>{rank_width}}  {warehouse_id:>{id_width}}"
            f"  {warehouse_items:>{item_width}}  {warehouse_fill:>{fill_width}}"
            f"     {algorithm_items:>{item_width}}  {algorithm_fill:>{fill_width}}"
        )

    print(sep)

    extra_w = len(w_containers) - len(a_containers)
    if extra_w > 0:
        print(f"  (Algorithm eliminated {extra_w} container(s) — shown as '—' above)")
    elif extra_w < 0:
        print(f"  (Algorithm required {-extra_w} extra container(s))")
    print()


def _quantity(row: dict[str, str]) -> int:
    return int(row["quantity"]) if row.get("quantity") else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    container_dims = parse_container_dims(args.container)
    if not container_dims:
        print(f"Error: could not parse --container {args.container!r}. Use LxWxH format.")
        return 1

    algorithm = AlgorithmConfig(mode=args.mode)
    comparison_config = ComparisonConfig(
        container_dims=container_dims,
        max_workers=args.workers,
        algorithm=algorithm,
    )

    print("Loading data...")
    item_dims = load_item_master()

    if args.list_jobs:
        all_events = load_picking_events()
        by_job: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in all_events:
            by_job[row["job"]].append(row)
        print(f"\n  {'Job ID':<22}  {'Containers':>10}  {'Pick lines':>10}  {'Items':>6}  {'Start time'}")
        print(f"  {'─' * 22}  {'─' * 10}  {'─' * 10}  {'─' * 6}  {'─' * 19}")
        for job, events in sorted(by_job.items(), key=lambda item: min(r["event_end"] for r in item[1])):
            containers = len({event["container"] for event in events})
            items = sum(_quantity(event) for event in events)
            start = min(event["event_end"] for event in events)[:19]
            print(f"  {job:<22}  {containers:>10}  {len(events):>10}  {items:>6}  {start}")
        print()
        return 0

    if args.job:
        events = load_picking_events(job=args.job, limit=args.limit)
        print(f"  Job: {args.job}")
        print(f"  Algorithm mode: {args.mode}")
        if args.limit is not None:
            print(f"  Limit: first {args.limit} pick lines")
        print(f"  {len(events)} picking events, {len(item_dims):,} SKUs in item master")

        if not events:
            print("No picking events found. Run with --list-jobs to see available job IDs.")
            return 0

        stop_timer = threading.Event()
        summary = {"n_cuboids": 0, "n_containers": 0}

        def live_timer() -> None:
            start = time.perf_counter()
            while not stop_timer.is_set():
                elapsed = time.perf_counter() - start
                print(f"  Elapsed: {elapsed:.1f}s", end="\r", flush=True)
                time.sleep(0.1)
            elapsed = time.perf_counter() - start
            print(
                f"  Done in {elapsed:.2f}s "
                f"({summary['n_cuboids']} items → {summary['n_containers']} container(s))"
            )

        timer_thread = threading.Thread(target=live_timer, daemon=True)
        timer_thread.start()
        result = compare_job(args.job, events, item_dims, comparison_config)
        summary["n_cuboids"] = int(result["n_cuboids"])
        summary["n_containers"] = int(result["als"]["total_containers"])
        stop_timer.set()
        timer_thread.join()

        ws = result["ws"]
        als = result["als"]
        w_containers = result["w_containers"]
        a_containers = result["a_containers"]
        assert isinstance(ws, dict)
        assert isinstance(als, dict)
        assert isinstance(w_containers, list)
        assert isinstance(a_containers, list)

        if ws["missing_dims"]:
            print(f"  Warning: {ws['missing_dims']} pick line(s) have no dimension data")
        if result["skipped"]:
            print(f"  Warning: {result['skipped']} pick line(s) skipped (SKU not in item master)")
        for warning in result["skip_warnings"]:
            print(warning)

        print_comparison(ws, als, container_dims)
        print_per_container_comparison(w_containers, a_containers, container_dims)
        return 0

    events_by_job = load_events_by_job(limit=args.limit)
    total_events = sum(len(events) for events in events_by_job.values())

    print(f"  Algorithm mode: {args.mode}")
    if args.limit is not None:
        print(f"  Limit: first {args.limit} pick lines")
    print(
        f"  {total_events} picking events across {len(events_by_job)} job(s), "
        f"{len(item_dims):,} SKUs in item master"
    )

    if not events_by_job:
        print("No picking events found.")
        return 0

    print(
        f"Running packing algorithm across {len(events_by_job)} job(s) "
        f"({comparison_config.max_workers or 'auto'} workers)..."
    )
    job_results = run_jobs_parallel(events_by_job, item_dims, comparison_config)

    for warning in [warning for result in job_results for warning in result["skip_warnings"]]:
        print(warning)

    ws_agg, als_agg = aggregate_across_jobs(job_results)
    print_comparison(ws_agg, als_agg, container_dims)
    print_per_job_comparison(job_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

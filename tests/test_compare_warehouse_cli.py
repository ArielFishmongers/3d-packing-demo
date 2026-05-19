from __future__ import annotations

import importlib.util
from pathlib import Path


def load_compare_warehouse_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "compare_warehouse.py"
    spec = importlib.util.spec_from_file_location("compare_warehouse_cli", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_job_results():
    return [
        {
            "job_id": "job-1",
            "ws": {
                "total_containers": 1,
                "total_items": 2,
                "total_lines": 1,
                "avg_items": 2.0,
                "min_items": 2,
                "max_items": 2,
                "avg_lines": 1.0,
                "avg_util": 50.0,
                "min_util": 50.0,
                "max_util": 50.0,
                "missing_dims": 0,
            },
            "als": {
                "total_containers": 1,
                "total_items": 2,
                "total_lines": None,
                "avg_items": 2.0,
                "min_items": 2,
                "max_items": 2,
                "avg_lines": None,
                "avg_util": 60.0,
                "min_util": 60.0,
                "max_util": 60.0,
                "missing_dims": 0,
            },
            "w_containers": [{"rank": 1, "container": "W1", "items": 2, "util_pct": 50.0}],
            "a_containers": [{"rank": 1, "items": 2, "util_pct": 60.0}],
            "skipped": 0,
            "n_cuboids": 2,
            "skip_warnings": [],
        }
    ]


def test_build_parser_supports_mode_flag():
    module = load_compare_warehouse_module()

    parser = module.build_parser()

    assert parser.parse_args([]).mode == "sequential"
    assert parser.parse_args(["--mode", "ga"]).mode == "ga"


def test_main_default_mode_matches_explicit_sequential(monkeypatch, capsys):
    module = load_compare_warehouse_module()
    job_results = make_job_results()
    calls: list[str] = []

    monkeypatch.setattr(module, "load_item_master", lambda: {("C1", "SKU1"): {"dims": (1.0, 1.0, 1.0), "volume": 1.0}})
    monkeypatch.setattr(
        module,
        "load_events_by_job",
        lambda limit=None: {
            "job-1": [
                {
                    "job": "job-1",
                    "container": "W1",
                    "event_end": "2026-03-13T10:00:00",
                    "quantity": "2",
                    "consignee": "C1",
                    "sku": "SKU1",
                }
            ]
        },
    )

    def fake_run_jobs_parallel(events_by_job, item_dims, config):
        calls.append(config.algorithm.mode)
        return job_results

    monkeypatch.setattr(module, "run_jobs_parallel", fake_run_jobs_parallel)

    exit_code_default = module.main(["--workers", "1"])
    output_default = capsys.readouterr().out
    exit_code_explicit = module.main(["--mode", "sequential", "--workers", "1"])
    output_explicit = capsys.readouterr().out

    assert exit_code_default == 0
    assert exit_code_explicit == 0
    assert calls == ["sequential", "sequential"]
    assert output_default == output_explicit

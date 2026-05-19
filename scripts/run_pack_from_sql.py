from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from bin_packing import Packer, PackerConfig, PackingResult
from bin_packing.io_postgres import QueryLoadResult, load_cuboids_from_sql_file

DEFAULT_CONTAINER_DIMS = (27, 50, 32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 3D packing algorithm against rows fetched from a PostgreSQL query.",
    )
    parser.add_argument(
        "--query-file",
        required=True,
        type=Path,
        help="Path to a .sql file that returns the item rows.",
    )
    parser.add_argument(
        "--mode",
        choices=("heuristic", "ga", "sequential"),
        default="heuristic",
        help="Packing mode to run.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=1,
        help="Voxel scale factor for decimal dimensions.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("volume", "longest_edge"),
        default="volume",
        help="Pre-sort key for heuristic and GA modes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed used by GA mode.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.precision < 1:
        parser.error("--precision must be at least 1.")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL environment variable is required.")

    try:
        load_result = load_cuboids_from_sql_file(database_url, args.query_file)
        result = run_packing_mode(
            load_result,
            mode=args.mode,
            precision=args.precision,
            sort_by=args.sort_by,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(
        load_result=load_result,
        result=result,
        mode=args.mode,
        container_dims=DEFAULT_CONTAINER_DIMS,
    )
    return 0


def run_packing_mode(
    load_result: QueryLoadResult,
    *,
    mode: str,
    precision: int,
    sort_by: str,
    seed: int | None,
) -> PackingResult:
    config = PackerConfig(
        use_ga=(mode == "ga"),
        sort_by=sort_by,
        precision=precision,
    )
    packer = Packer(DEFAULT_CONTAINER_DIMS, config=config, seed=seed)

    if mode == "sequential":
        return packer.pack_sequential(load_result.cuboids)
    return packer.pack(load_result.cuboids)


def print_summary(
    *,
    load_result: QueryLoadResult,
    result: PackingResult,
    mode: str,
    container_dims: tuple[int, int, int],
) -> None:
    print(f"Mode: {mode}")
    print(f"Container dims: {container_dims}")
    print(f"Fetched rows: {load_result.row_count}")
    print(f"Expanded items: {len(load_result.cuboids)}")
    print(f"Packed items: {len(result.placements)}")
    print(f"Unpacked items: {len(result.unpacked_ids)}")
    print(f"Utilisation: {result.utilisation:.1%}")
    print(f"Max height: {result.h_max}")

    if result.unpacked_ids:
        print("Unpacked IDs:")
        for item_id in result.unpacked_ids:
            print(f"  - {item_id}")

    print("Placements:")
    for placement in result.placements:
        x, y, z = placement.position
        l, w, h = placement.orientation.dims
        print(
            f"  - {placement.item_id}: pos=({x}, {y}, {z}) dims=({l}, {w}, {h})"
        )


if __name__ == "__main__":
    raise SystemExit(main())

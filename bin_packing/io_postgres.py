from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Cuboid

REQUIRED_COLUMNS = ("item_id", "length", "width", "height")


@dataclass(frozen=True)
class QueryLoadResult:
    row_count: int
    cuboids: list[Cuboid]


@dataclass(frozen=True)
class _ItemSpec:
    item_id: str
    dims: tuple[int, int, int]
    rotations_allowed: bool
    weight: float
    quantity: int


def read_query_file(path: str | Path) -> str:
    query = Path(path).read_text(encoding="utf-8")
    if not query.strip():
        raise ValueError("SQL query file is empty.")
    return query


def execute_query(database_url: str, query: str) -> list[Mapping[str, Any]]:
    if not database_url:
        raise ValueError("DATABASE_URL is required.")
    if not query.strip():
        raise ValueError("SQL query is empty.")

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is required for PostgreSQL import. Install with `pip install -e .[postgres]`."
        ) from exc

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return list(cur.fetchall())


def load_cuboids_from_sql_file(database_url: str, query_file: str | Path) -> QueryLoadResult:
    query = read_query_file(query_file)
    rows = execute_query(database_url, query)
    cuboids = rows_to_cuboids(rows)
    return QueryLoadResult(row_count=len(rows), cuboids=cuboids)


def rows_to_cuboids(rows: Iterable[Mapping[str, Any]]) -> list[Cuboid]:
    specs = [_row_to_item_spec(row, idx) for idx, row in enumerate(rows, start=1)]
    total_counts = Counter()
    for spec in specs:
        total_counts[spec.item_id] += spec.quantity

    emitted_counts: Counter[str] = Counter()
    cuboids: list[Cuboid] = []
    for spec in specs:
        for _ in range(spec.quantity):
            emitted_counts[spec.item_id] += 1
            item_id = _resolved_item_id(
                spec.item_id,
                emitted_counts[spec.item_id],
                total_counts[spec.item_id],
            )
            cuboids.append(
                Cuboid(
                    item_id=item_id,
                    dims=spec.dims,
                    weight=spec.weight,
                    rotations_allowed=spec.rotations_allowed,
                )
            )
    return cuboids


def _resolved_item_id(base_item_id: str, item_number: int, total_count: int) -> str:
    if total_count == 1:
        return base_item_id
    return f"{base_item_id}_{item_number}"


def _row_to_item_spec(row: Mapping[str, Any], row_number: int) -> _ItemSpec:
    missing = [column for column in REQUIRED_COLUMNS if column not in row]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Row {row_number} is missing required column(s): {missing_list}.")

    raw_item_id = row["item_id"]
    if raw_item_id is None:
        raise ValueError(f"Row {row_number} has null item_id.")
    item_id = str(raw_item_id).strip()
    if not item_id:
        raise ValueError(f"Row {row_number} has an empty item_id.")

    dims = (
        _coerce_positive_int(row["length"], "length", row_number),
        _coerce_positive_int(row["width"], "width", row_number),
        _coerce_positive_int(row["height"], "height", row_number),
    )
    rotations_allowed = _coerce_bool(
        row.get("rotations_allowed", True),
        "rotations_allowed",
        row_number,
    )
    weight = _coerce_float(row.get("weight", 1.0), "weight", row_number)
    quantity = _coerce_positive_int(row.get("quantity", 1), "quantity", row_number)

    return _ItemSpec(
        item_id=item_id,
        dims=dims,
        rotations_allowed=rotations_allowed,
        weight=weight,
        quantity=quantity,
    )


def _coerce_positive_int(value: Any, column: str, row_number: int) -> int:
    if value is None:
        raise ValueError(f"Row {row_number} column {column!r} cannot be null.")

    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Row {row_number} column {column!r} must be an integer-compatible value."
        ) from exc

    if value != coerced and not (isinstance(value, str) and value.strip() == str(coerced)):
        raise ValueError(f"Row {row_number} column {column!r} must be an integer value.")
    if coerced <= 0:
        raise ValueError(f"Row {row_number} column {column!r} must be positive.")
    return coerced


def _coerce_float(value: Any, column: str, row_number: int) -> float:
    if value is None:
        raise ValueError(f"Row {row_number} column {column!r} cannot be null.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Row {row_number} column {column!r} must be numeric.") from exc


def _coerce_bool(value: Any, column: str, row_number: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y"}:
            return True
        if lowered in {"0", "false", "f", "no", "n"}:
            return False
    raise ValueError(
        f"Row {row_number} column {column!r} must be boolean-like (true/false or 1/0)."
    )

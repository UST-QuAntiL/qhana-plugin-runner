from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Iterable

import duckdb

PREVIEW_LIMIT = 100
ALLOWED_STATEMENT_TYPES = {duckdb.StatementType.SELECT}
READ_ONLY_PRAGMAS = {
    "show_tables",
    "show_databases",
    "show_schemas",
    "table_info",
}
PRAGMA_PATTERN = re.compile(r"^pragma\s+([a-z_]+)\b", re.IGNORECASE)


def normalize_sql(sql: str) -> str:
    """Normalize SQL input by trimming whitespace and trailing semicolons."""
    normalized = (sql or "").strip()
    while normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    return normalized


def strip_leading_comments(sql: str) -> str:
    """Remove leading line/block comments to inspect the actual statement."""
    stripped = sql.lstrip()
    while True:
        if stripped.startswith("--"):
            newline = stripped.find("\n")
            if newline == -1:
                return ""
            stripped = stripped[newline + 1 :].lstrip()
            continue
        if stripped.startswith("/*"):
            end = stripped.find("*/")
            if end == -1:
                return ""
            stripped = stripped[end + 2 :].lstrip()
            continue
        return stripped


# Security hardening without restricting access in duckdb
# TBD whether enough or not
def validate_pragma(sql: str) -> str | None:
    """Allow only whitelisted, read-only PRAGMA statements."""
    match = PRAGMA_PATTERN.match(sql)
    if not match:
        return None
    pragma_name = match.group(1).lower()
    if pragma_name not in READ_ONLY_PRAGMAS:
        return f"PRAGMA '{pragma_name}' is not allowed."
    return None


def validate_sql(sql: str) -> tuple[str | None, str]:
    """Parse SQL and enforce single-statement, read-only policy."""
    normalized = normalize_sql(sql)
    if not normalized:
        return "SQL query is required.", normalized
    pragma_error = validate_pragma(strip_leading_comments(normalized))
    if pragma_error:
        return pragma_error, normalized
    try:
        statements = duckdb.extract_statements(normalized)
    except duckdb.ParserException as err:
        return err.args[0], normalized
    if len(statements) != 1:
        return "Only a single SQL statement is allowed.", normalized
    statement = statements[0]
    if statement.type not in ALLOWED_STATEMENT_TYPES:
        return "Only SELECT statements are allowed.", normalized
    return None, normalized


def check_sql_syntax(sql: str) -> str | None:
    """Return a user-facing error message for invalid SQL, else None."""
    error, _ = validate_sql(sql)
    return error


def _prepare_connection(con: duckdb.DuckDBPyConnection) -> None:
    """Enable HTTP(S) reads by loading DuckDB's httpfs extension."""
    try:
        con.install_extension("httpfs")
    except duckdb.Error:
        # Extension may already be installed or installation may be unavailable.
        pass
    try:
        con.load_extension("httpfs")
    except duckdb.Error as err:
        raise RuntimeError(
            "Failed to load DuckDB httpfs extension required for HTTP(S) sources."
        ) from err


def execute_sql(sql: str, *, limit: int | None = None) -> tuple[list[str], list[tuple]]:
    """Run validated SQL and return column names plus result rows."""
    error, normalized = validate_sql(sql)
    if error:
        raise ValueError(error)
    with duckdb.connect(
        config={
            "python_enable_replacements": False,
            "allow_community_extensions": False,
            "allow_persistent_secrets": False,
            # "disabled_filesystems": "LocalFileSystem",
        }
    ) as con:
        # FIXME: https://duckdb.org/docs/stable/operations_manual/
        # securing_duckdb/overview
        _prepare_connection(con)
        relation = con.sql(normalized)
        if limit is not None:
            relation = relation.limit(limit)
        columns = relation.columns
        rows = relation.fetchall()
    return columns, rows


def serialize_value(value):
    """Convert result values into JSON-safe primitives."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def serialize_rows(rows: Iterable[tuple]) -> list[list]:
    """Serialize an iterable of rows into JSON-friendly lists."""
    return [[serialize_value(value) for value in row] for row in rows]


def rows_to_records(columns: list[str], rows: Iterable[tuple]) -> list[dict]:
    """Map result rows to dict records keyed by column name."""
    return [
        {column: serialize_value(value) for column, value in zip(columns, row)}
        for row in rows
    ]

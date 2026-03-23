"""DuckDB setup and schema management."""

import json
import uuid
from typing import Any

import duckdb


def get_connection(path: str = "eval_loop.duckdb") -> duckdb.DuckDBPyConnection:
    return duckdb.connect(path)


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id          VARCHAR PRIMARY KEY,
            start_time  TIMESTAMP NOT NULL,
            end_time    TIMESTAMP,
            episodes_tested  INTEGER,
            final_avg_score  DOUBLE,
            total_cost_usd   DOUBLE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS iterations (
            id            VARCHAR PRIMARY KEY,
            episode_id    VARCHAR NOT NULL,
            run_id        VARCHAR NOT NULL REFERENCES runs(id),
            iteration     INTEGER NOT NULL,
            system_prompt TEXT,
            task_output   TEXT,
            critique      TEXT,
            score         DOUBLE,
            cost_usd      DOUBLE,
            latency_ms    INTEGER,
            timestamp     TIMESTAMP NOT NULL DEFAULT current_timestamp
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            id         VARCHAR PRIMARY KEY,
            run_id     VARCHAR NOT NULL REFERENCES runs(id),
            iteration  INTEGER NOT NULL,
            kind       VARCHAR NOT NULL,
            message    TEXT NOT NULL,
            details    TEXT,
            timestamp  TIMESTAMP NOT NULL DEFAULT current_timestamp
        )
    """)


def log_anomaly(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    iteration: int,
    kind: str,
    message: str,
    details: Any,
) -> None:
    conn.execute(
        "INSERT INTO anomalies (id, run_id, iteration, kind, message, details) VALUES (?,?,?,?,?,?)",
        [str(uuid.uuid4()), run_id, iteration, kind, message, json.dumps(details)],
    )

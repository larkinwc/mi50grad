"""
Autotuner for kernel configurations.

Searches over tile sizes, workgroup sizes, unroll factors, and LDS allocation
for each GEMM shape. Results stored in SQLite for reuse.

This is a Phase 4 component — stub for now.
"""

import sqlite3
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TuneConfig:
    """A single kernel configuration to benchmark."""
    tile_m: int
    tile_n: int
    tile_k: int
    block_size: int
    unroll_factor: int
    lds_bytes: int
    vgprs: int

    def key(self) -> str:
        return f"{self.tile_m}x{self.tile_n}x{self.tile_k}_b{self.block_size}_u{self.unroll_factor}"


@dataclass
class TuneResult:
    """Result of benchmarking a configuration."""
    config: TuneConfig
    time_ms: float
    tflops: float
    efficiency: float  # fraction of peak


class TuneDB:
    """SQLite database for storing tuning results."""

    def __init__(self, db_path: str = "tune.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tune_results (
                op TEXT NOT NULL,
                m INTEGER NOT NULL,
                n INTEGER NOT NULL,
                k INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                config TEXT NOT NULL,
                time_ms REAL NOT NULL,
                tflops REAL NOT NULL,
                efficiency REAL NOT NULL,
                timestamp REAL NOT NULL,
                PRIMARY KEY (op, m, n, k, dtype, config)
            )
        """)
        self.conn.commit()

    def store(self, op: str, m: int, n: int, k: int, dtype: str,
              result: TuneResult):
        self.conn.execute(
            "INSERT OR REPLACE INTO tune_results VALUES (?,?,?,?,?,?,?,?,?,?)",
            (op, m, n, k, dtype, result.config.key(),
             result.time_ms, result.tflops, result.efficiency, time.time())
        )
        self.conn.commit()

    def best(self, op: str, m: int, n: int, k: int, dtype: str) -> Optional[TuneResult]:
        row = self.conn.execute(
            "SELECT config, time_ms, tflops, efficiency FROM tune_results "
            "WHERE op=? AND m=? AND n=? AND k=? AND dtype=? "
            "ORDER BY time_ms ASC LIMIT 1",
            (op, m, n, k, dtype)
        ).fetchone()
        if row is None:
            return None
        # Return just the time/perf — config reconstruction happens at kernel level
        return TuneResult(
            config=TuneConfig(0, 0, 0, 0, 0, 0, 0),  # placeholder
            time_ms=row[1], tflops=row[2], efficiency=row[3]
        )

    def close(self):
        self.conn.close()

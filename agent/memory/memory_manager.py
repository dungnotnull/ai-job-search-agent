"""
memory_manager.py — Persistent memory for ai-job-search-enhanced.
SQLite backend: search_sessions, salary_cache, llm_cost_log, knowledge_hashes, sentiment_cache.
"""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "agent_memory.db"


class MemoryManager:
    """Thread-safe SQLite memory manager for the career intelligence agent."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS search_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT,
                    n_jobs_fetched INTEGER,
                    top5_match_avg REAL,
                    gap_skills_count INTEGER,
                    letters_generated INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS salary_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    features_hash TEXT UNIQUE NOT NULL,
                    low INTEGER,
                    median INTEGER,
                    high INTEGER,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_key TEXT UNIQUE NOT NULL,
                    culture REAL,
                    management REAL,
                    wlb REAL,
                    overall REAL,
                    label TEXT,
                    review_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS llm_cost_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT,
                    model TEXT,
                    use_case TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost_usd REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS knowledge_hashes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT UNIQUE NOT NULL,
                    title TEXT,
                    source TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def save_session(self, session_id: str, data: dict[str, Any]):
        top5 = data.get("top5_matches", [])
        avg_score = sum(top5) / len(top5) if top5 else 0.0

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO search_sessions
                       (session_id, query, n_jobs_fetched, top5_match_avg, gap_skills_count, letters_generated)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        data.get("query", ""),
                        data.get("n_jobs_fetched", 0),
                        round(avg_score, 4),
                        data.get("gap_skills_count", 0),
                        data.get("letters_generated", 0),
                    ),
                )

    def get_session_stats(self) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as total, AVG(top5_match_avg) as avg_match "
                "FROM search_sessions"
            ).fetchone()
            return {
                "total_sessions": row["total"],
                "avg_top5_match": round(row["avg_match"] or 0.0, 4),
            }

    def get_salary_cache(self, features: dict) -> Optional[dict]:
        key = hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM salary_cache WHERE features_hash = ?", (key,)
            ).fetchone()
            if row:
                return {"low": row["low"], "median": row["median"],
                        "high": row["high"], "confidence": row["confidence"]}
        return None

    def save_salary_cache(self, features: dict, prediction):
        key = hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO salary_cache
                       (features_hash, low, median, high, confidence)
                       VALUES (?, ?, ?, ?, ?)""",
                    (key, prediction.low, prediction.median, prediction.high, prediction.confidence),
                )

    def get_sentiment_cache(self, company: str) -> Optional[dict]:
        key = company.lower().strip()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sentiment_cache WHERE company_key = ?", (key,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def save_sentiment_cache(self, company: str, sentiment):
        key = company.lower().strip()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO sentiment_cache
                       (company_key, culture, management, wlb, overall, label, review_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (key, sentiment.culture, sentiment.management, sentiment.wlb,
                     sentiment.overall, sentiment.label, sentiment.review_count),
                )

    def log_llm_cost(
        self,
        provider: str,
        model: str,
        use_case: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ):
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO llm_cost_log
                       (provider, model, use_case, input_tokens, output_tokens, cost_usd)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (provider, model, use_case, input_tokens, output_tokens, cost_usd),
                )

    def get_cost_summary(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT provider, model, use_case,
                          COUNT(*) as calls,
                          SUM(input_tokens) as total_input,
                          SUM(output_tokens) as total_output,
                          SUM(cost_usd) as total_cost
                   FROM llm_cost_log
                   GROUP BY provider, model, use_case
                   ORDER BY total_cost DESC"""
            ).fetchall()
            total = conn.execute("SELECT SUM(cost_usd) as t FROM llm_cost_log").fetchone()
            return {
                "total_cost_usd": round(total["t"] or 0.0, 6),
                "by_provider_model_usecase": [dict(r) for r in rows],
            }

    def is_known_paper(self, title: str, source: str = "") -> bool:
        h = hashlib.sha256(f"{title.lower()}{source}".encode()).hexdigest()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM knowledge_hashes WHERE content_hash = ?", (h,)
            ).fetchone()
            return row is not None

    def mark_paper_known(self, title: str, source: str = ""):
        h = hashlib.sha256(f"{title.lower()}{source}".encode()).hexdigest()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO knowledge_hashes (content_hash, title, source) VALUES (?, ?, ?)",
                    (h, title[:500], source[:500]),
                )

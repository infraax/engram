"""
kvcos/engram/index_c.py — Confidence history index for ENGRAM.

Stores retrieval confidence records across sessions.
Makes the system self-aware: chronic failures are known before retrieval.

Schema:
  retrievals:     one row per geodesic_retrieve() call
  confusion_pairs: doc pairs that confuse each other (confidence<threshold)
  doc_stats:      per-doc aggregate reliability scores

Usage:
  ic = IndexC.open("results/index_c.db")
  ic.record(session_id="s1", query_doc_id="doc_146", result=geodesic_result)
  prior = ic.prior("doc_146")
  pairs = ic.confusion_registry()
  rmap  = ic.reliability_map()
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConfidenceRecord:
    session_id: str
    query_doc_id: str
    result_doc_id: str
    confidence: str
    margin: float
    stages_used: int
    constraint_used: bool
    correct: bool
    ts: float


@dataclass
class DocPrior:
    """Prior confidence distribution for a doc_id."""

    doc_id: str
    n_high: int
    n_medium: int
    n_low: int
    n_total: int
    reliability: float
    is_chronic_failure: bool

    @property
    def dominant_confidence(self) -> str:
        if self.n_total == 0:
            return "unknown"
        counts = {
            "high": self.n_high,
            "medium": self.n_medium,
            "low": self.n_low,
        }
        return max(counts, key=counts.get)


@dataclass
class ConfusionPair:
    doc_a: str
    doc_b: str
    n_confusions: int
    first_seen: float
    last_seen: float


SCHEMA = """
CREATE TABLE IF NOT EXISTS retrievals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    query_doc_id    TEXT    NOT NULL,
    result_doc_id   TEXT    NOT NULL,
    confidence      TEXT    NOT NULL,
    margin          REAL    NOT NULL,
    stages_used     INTEGER NOT NULL,
    constraint_used INTEGER NOT NULL,
    correct         INTEGER NOT NULL,
    ts              REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ret_query  ON retrievals(query_doc_id);
CREATE INDEX IF NOT EXISTS idx_ret_result ON retrievals(result_doc_id);
CREATE INDEX IF NOT EXISTS idx_ret_conf   ON retrievals(confidence);
CREATE INDEX IF NOT EXISTS idx_ret_sess   ON retrievals(session_id);

CREATE TABLE IF NOT EXISTS confusion_pairs (
    doc_a         TEXT NOT NULL,
    doc_b         TEXT NOT NULL,
    n_confusions  INTEGER NOT NULL DEFAULT 1,
    first_seen    REAL    NOT NULL,
    last_seen     REAL    NOT NULL,
    PRIMARY KEY (doc_a, doc_b)
);

CREATE TABLE IF NOT EXISTS doc_stats (
    doc_id          TEXT    PRIMARY KEY,
    n_high          INTEGER NOT NULL DEFAULT 0,
    n_medium        INTEGER NOT NULL DEFAULT 0,
    n_low           INTEGER NOT NULL DEFAULT 0,
    reliability     REAL    NOT NULL DEFAULT 1.0,
    last_updated    REAL    NOT NULL
);
"""


class IndexC:
    """
    Confidence history index.

    Backed by SQLite — append-only, persistent across sessions.
    Provides priors for geodesic_retrieve() to pre-apply constraints
    on docs that are known chronic failures.
    """

    CHRONIC_FAILURE_THRESHOLD = 0.5

    def __init__(self, db_path: str):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    @classmethod
    def open(cls, db_path: str | Path) -> "IndexC":
        """Open (or create) the Index-C database at db_path."""
        os.makedirs(Path(db_path).parent, exist_ok=True)
        return cls(str(db_path))

    # ── WRITE ────────────────────────────────────────────────────────

    def record(
        self,
        session_id: str,
        query_doc_id: str,
        result_doc_id: str,
        confidence: str,
        margin: float,
        stages_used: int = 1,
        constraint_used: bool = False,
        correct: bool = True,
        ts: float | None = None,
    ) -> None:
        """Log one retrieval result to the index."""
        ts = ts or time.time()
        with self._conn:
            self._conn.execute(
                """INSERT INTO retrievals
                   (session_id, query_doc_id, result_doc_id, confidence,
                    margin, stages_used, constraint_used, correct, ts)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (session_id, query_doc_id, result_doc_id, confidence,
                 float(margin), int(stages_used), int(constraint_used),
                 int(correct), float(ts)),
            )
            if not correct:
                self._register_confusion(query_doc_id, result_doc_id, ts)
            self._update_doc_stats(query_doc_id, confidence, ts)

    def _register_confusion(
        self, doc_a: str, doc_b: str, ts: float
    ) -> None:
        """Insert or increment confusion pair."""
        existing = self._conn.execute(
            "SELECT n_confusions FROM confusion_pairs "
            "WHERE doc_a=? AND doc_b=?",
            (doc_a, doc_b),
        ).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE confusion_pairs SET n_confusions=n_confusions+1, "
                "last_seen=? WHERE doc_a=? AND doc_b=?",
                (ts, doc_a, doc_b),
            )
        else:
            self._conn.execute(
                "INSERT INTO confusion_pairs "
                "(doc_a, doc_b, n_confusions, first_seen, last_seen) "
                "VALUES (?,?,1,?,?)",
                (doc_a, doc_b, ts, ts),
            )

    def _update_doc_stats(
        self, doc_id: str, confidence: str, ts: float
    ) -> None:
        """Upsert doc_stats row for doc_id."""
        col_map = {"high": "n_high", "medium": "n_medium", "low": "n_low"}
        col = col_map.get(confidence, "n_medium")

        existing = self._conn.execute(
            "SELECT n_high, n_medium, n_low FROM doc_stats WHERE doc_id=?",
            (doc_id,),
        ).fetchone()

        if existing:
            n_high, n_medium, n_low = existing
            if col == "n_high":
                n_high += 1
            elif col == "n_medium":
                n_medium += 1
            else:
                n_low += 1
            n_total = n_high + n_medium + n_low
            reliability = (n_high + n_medium) / n_total if n_total > 0 else 1.0
            self._conn.execute(
                "UPDATE doc_stats SET n_high=?, n_medium=?, n_low=?, "
                "reliability=?, last_updated=? WHERE doc_id=?",
                (n_high, n_medium, n_low, reliability, ts, doc_id),
            )
        else:
            vals = {"n_high": 0, "n_medium": 0, "n_low": 0}
            vals[col] = 1
            reliability = (vals["n_high"] + vals["n_medium"]) / 1
            self._conn.execute(
                "INSERT INTO doc_stats "
                "(doc_id, n_high, n_medium, n_low, reliability, last_updated) "
                "VALUES (?,?,?,?,?,?)",
                (doc_id, vals["n_high"], vals["n_medium"],
                 vals["n_low"], reliability, ts),
            )

    # ── READ ─────────────────────────────────────────────────────────

    def prior(self, doc_id: str) -> DocPrior:
        """Return prior confidence distribution for doc_id."""
        row = self._conn.execute(
            "SELECT n_high, n_medium, n_low, reliability "
            "FROM doc_stats WHERE doc_id=?",
            (doc_id,),
        ).fetchone()
        if not row:
            return DocPrior(
                doc_id=doc_id, n_high=0, n_medium=0, n_low=0,
                n_total=0, reliability=1.0, is_chronic_failure=False,
            )
        n_high, n_medium, n_low, reliability = row
        n_total = n_high + n_medium + n_low
        return DocPrior(
            doc_id=doc_id,
            n_high=n_high,
            n_medium=n_medium,
            n_low=n_low,
            n_total=n_total,
            reliability=reliability,
            is_chronic_failure=(
                n_low / n_total > self.CHRONIC_FAILURE_THRESHOLD
                if n_total > 0
                else False
            ),
        )

    def confusion_registry(
        self, min_confusions: int = 1
    ) -> list[ConfusionPair]:
        """Return known confusion pairs with >= min_confusions."""
        rows = self._conn.execute(
            "SELECT doc_a, doc_b, n_confusions, first_seen, last_seen "
            "FROM confusion_pairs WHERE n_confusions >= ? "
            "ORDER BY n_confusions DESC",
            (min_confusions,),
        ).fetchall()
        return [
            ConfusionPair(
                doc_a=r[0], doc_b=r[1], n_confusions=r[2],
                first_seen=r[3], last_seen=r[4],
            )
            for r in rows
        ]

    def reliability_map(self) -> dict[str, float]:
        """Return {doc_id: reliability_score} for all tracked docs."""
        rows = self._conn.execute(
            "SELECT doc_id, reliability FROM doc_stats ORDER BY reliability"
        ).fetchall()
        return {r[0]: float(r[1]) for r in rows}

    def session_history(self, session_id: str) -> list[ConfidenceRecord]:
        """Return all records for a session_id."""
        rows = self._conn.execute(
            "SELECT session_id, query_doc_id, result_doc_id, confidence, "
            "margin, stages_used, constraint_used, correct, ts "
            "FROM retrievals WHERE session_id=? ORDER BY ts",
            (session_id,),
        ).fetchall()
        return [
            ConfidenceRecord(
                session_id=r[0], query_doc_id=r[1], result_doc_id=r[2],
                confidence=r[3], margin=float(r[4]), stages_used=int(r[5]),
                constraint_used=bool(r[6]), correct=bool(r[7]), ts=float(r[8]),
            )
            for r in rows
        ]

    def n_sessions(self) -> int:
        """Number of distinct sessions recorded."""
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM retrievals"
        ).fetchone()
        return row[0] if row else 0



    # ── RECENCY-WEIGHTED RELIABILITY ────────────────────────────────

    def weighted_reliability(
        self,
        doc_id: str,
        decay:  float = 0.85,
    ) -> float:
        """
        Exponentially weighted reliability score.
        Newer retrievals have higher weight.
        decay=0.85: a failure 5 sessions ago counts 0.85^5 = 0.44
        of a failure last session.

        Returns float in [0, 1]. Returns 1.0 if no history.
        """
        rows = self._conn.execute(
            """SELECT correct FROM retrievals
               WHERE query_doc_id=?
               ORDER BY ts ASC""",
            (doc_id,),
        ).fetchall()
        if not rows:
            return 1.0
        history = [bool(r[0]) for r in rows]
        n       = len(history)
        weights = [decay ** (n - 1 - i) for i in range(n)]
        total_w = sum(weights)
        score   = sum(w * int(h) for w, h in zip(weights, history))
        return round(score / total_w, 6) if total_w > 0 else 1.0

    # ── INDEX GROWTH REVALIDATION ────────────────────────────────────

    def on_document_added(
        self,
        new_doc_id:           str,
        new_vec,              # torch.Tensor [dim]
        hnsw_index,           # EngramIndex instance
        revalidation_radius:  float = 0.85,
        density_threshold:    int   = 3,
    ) -> list[str]:
        """
        Call after adding a document to the HNSW index.
        Recomputes local_density for neighbors of new_doc_id
        whose similarity > revalidation_radius.

        Returns list of doc_ids whose density was updated.

        Why this matters:
          At N=200, doc_042 is in sparse space (density=1).
          At N=500, doc_201 lands near doc_042 (cosine=0.88).
          doc_042 is now in a denser region — its confidence tier
          has degraded. This method detects and records that.
        """
        import torch
        import torch.nn.functional as F

        updated = []
        try:
            results = hnsw_index.search(new_vec, top_k=20)
        except Exception:
            return updated

        for r in results:
            if r.doc_id == new_doc_id:
                continue
            if r.score < revalidation_radius:
                continue

            # Recompute density for this neighbor
            neighbor_vec = hnsw_index.get_vector(r.doc_id)
            if neighbor_vec is None:
                continue

            all_results = hnsw_index.search(neighbor_vec, top_k=50)
            new_density = sum(
                1 for x in all_results
                if x.doc_id != r.doc_id and x.score > revalidation_radius
            )

            ts = __import__("time").time()
            existing = self._conn.execute(
                "SELECT n_high FROM doc_stats WHERE doc_id=?",
                (r.doc_id,),
            ).fetchone()

            if existing:
                self._conn.execute(
                    "UPDATE doc_stats SET last_updated=? WHERE doc_id=?",
                    (ts, r.doc_id),
                )
            else:
                self._conn.execute(
                    "INSERT OR IGNORE INTO doc_stats "
                    "(doc_id, n_high, n_medium, n_low, reliability, last_updated) "
                    "VALUES (?,0,0,0,1.0,?)",
                    (r.doc_id, ts),
                )

            # If density crossed threshold, register as needing constraint activation
            if new_density > density_threshold:
                self._register_confusion(r.doc_id, new_doc_id, ts)

            updated.append(r.doc_id)
            self._conn.commit()

        return updated

    def close(self) -> None:
        self._conn.close()

    def __repr__(self) -> str:
        n = self._conn.execute(
            "SELECT COUNT(*) FROM retrievals"
        ).fetchone()[0]
        return f"IndexC(db={self._db_path!r}, n_records={n})"

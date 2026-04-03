"""
kvcos/engram/session_propagator.py — Session start/end for ENGRAM.

Bridges geodesic_retrieve() results and IndexC persistence.
Call session_start() at the top of each session to load priors.
Call session_end() at the bottom to persist results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from kvcos.engram.index_c import IndexC, DocPrior


@dataclass
class SessionSummary:
    session_id: str
    n_total: int
    n_correct: int
    n_high: int
    n_medium: int
    n_low: int
    n_preempted: int
    new_confusion_pairs: list[tuple[str, str]]
    recall: float
    duration_s: float


class SessionPropagator:
    """
    Manages session-level IndexC writes.

    Accumulates retrieval results in memory during the session.
    Writes them to IndexC at session_end().
    """

    def __init__(self, db_path: str, session_id: str):
        self._db_path = str(db_path)
        self._session_id = session_id
        self._ic: IndexC | None = None
        self._records: list[dict] = []
        self._start_ts: float = 0.0
        self._started: bool = False

    def session_start(self) -> dict[str, DocPrior]:
        """
        Open IndexC, return {doc_id: DocPrior} for all known docs.
        Call at the top of each session.
        """
        self._ic = IndexC.open(self._db_path)
        self._start_ts = time.time()
        self._started = True

        rmap = self._ic.reliability_map()
        return {
            doc_id: self._ic.prior(doc_id)
            for doc_id in rmap
        }

    @property
    def index_c(self) -> IndexC:
        """Access the IndexC instance (after session_start)."""
        if self._ic is None:
            raise RuntimeError("Call session_start() first.")
        return self._ic

    def record(
        self,
        query_doc_id: str,
        result_doc_id: str,
        confidence: str,
        margin: float,
        stages_used: int = 1,
        constraint_used: bool = False,
        correct: bool = True,
    ) -> None:
        """Buffer one retrieval result for this session."""
        self._records.append({
            "query_doc_id": query_doc_id,
            "result_doc_id": result_doc_id,
            "confidence": confidence,
            "margin": float(margin),
            "stages_used": int(stages_used),
            "constraint_used": bool(constraint_used),
            "correct": bool(correct),
            "ts": time.time(),
        })

    def session_end(self) -> SessionSummary:
        """Write all buffered records to IndexC. Return summary."""
        if not self._started or self._ic is None:
            raise RuntimeError("Call session_start() before session_end().")

        confusion_before = {
            (p.doc_a, p.doc_b)
            for p in self._ic.confusion_registry(min_confusions=1)
        }

        for rec in self._records:
            self._ic.record(
                session_id=self._session_id,
                query_doc_id=rec["query_doc_id"],
                result_doc_id=rec["result_doc_id"],
                confidence=rec["confidence"],
                margin=rec["margin"],
                stages_used=rec["stages_used"],
                constraint_used=rec["constraint_used"],
                correct=rec["correct"],
                ts=rec["ts"],
            )

        confusion_after = {
            (p.doc_a, p.doc_b)
            for p in self._ic.confusion_registry(min_confusions=1)
        }
        new_pairs = list(confusion_after - confusion_before)

        n_total = len(self._records)
        n_correct = sum(1 for r in self._records if r["correct"])
        counters = {"high": 0, "medium": 0, "low": 0}
        n_preempted = 0
        for r in self._records:
            counters[r["confidence"]] = counters.get(r["confidence"], 0) + 1
            if r["stages_used"] == 0:
                n_preempted += 1

        summary = SessionSummary(
            session_id=self._session_id,
            n_total=n_total,
            n_correct=n_correct,
            n_high=counters["high"],
            n_medium=counters["medium"],
            n_low=counters["low"],
            n_preempted=n_preempted,
            new_confusion_pairs=new_pairs,
            recall=n_correct / n_total if n_total > 0 else 0.0,
            duration_s=time.time() - self._start_ts,
        )

        self._ic.close()
        self._ic = None
        self._started = False
        self._records = []

        return summary

    def summary_str(self, s: SessionSummary) -> str:
        return (
            f"Session {s.session_id}: "
            f"{s.n_total} retrievals | "
            f"recall={s.recall:.1%} | "
            f"H={s.n_high}/M={s.n_medium}/L={s.n_low} | "
            f"preempted={s.n_preempted} | "
            f"new_pairs={len(s.new_confusion_pairs)} | "
            f"{s.duration_s:.1f}s"
        )

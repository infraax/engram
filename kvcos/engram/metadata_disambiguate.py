"""
kvcos/engram/metadata_disambiguate.py

Stage 4 retrieval: activates when the fingerprint pipeline returns LOW.
Uses .eng metadata fields (domain, context_len, l2_norm, task_description)
to break ties that the Fourier fingerprint cannot resolve.

Returns Stage4Result with confidence='low-metadata' and metadata_used=True.

Design note (VRCM source):
  When constraint satisfaction fails on the fingerprint axis, switch to
  orthogonal axes — metadata fields that are independent of spectral structure.
  The medicine/biology failure exists because their f0+f1 profiles are
  spectrally identical. Their metadata is NOT identical: context_len differs,
  l2_norm differs, task_description keywords differ. That orthogonal signal
  is what Stage 4 exploits.
"""

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class Stage4Result:
    doc_id:           str
    meta_score:       float
    confidence:       str   = 'low-metadata'
    metadata_used:    bool  = True
    domain_matched:   bool  = False
    score_breakdown:  dict  = None

    def __post_init__(self):
        if self.score_breakdown is None:
            self.score_breakdown = {}


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Jaccard overlap on lowercase word sets, excluding stopwords."""
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'of',
        'to', 'and', 'or', 'that', 'this', 'it', 'with', 'for',
        'on', 'at', 'by', 'from', 'be', 'has', 'have', 'had',
    }
    def words(t):
        return set(w for w in re.sub(r'[^a-z0-9 ]', ' ',
                                     (t or '').lower()).split()
                   if w not in STOPWORDS and len(w) > 2)
    a, b = words(text_a), words(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def metadata_disambiguate(
    candidates: list[dict],
    query_metadata: dict,
    domain_bonus:   float = 0.3,
    max_len:        int   = 8192,
    max_norm:       float = 10.0,
) -> Stage4Result | None:
    """
    Stage 4 disambiguation using .eng metadata fields.

    Args:
        candidates:      list of dicts from eng_index values.
                         Each must have: cache_id, task_description,
                         context_len (optional), l2_norm (optional),
                         metadata dict with domain (optional).
        query_metadata:  dict with same structure as one candidate.
        domain_bonus:    score added for exact domain match (default 0.3).
        max_len:         normalisation constant for context_len diff.
        max_norm:        normalisation constant for l2_norm diff.

    Returns:
        Stage4Result for the highest meta-scoring candidate, or None
        if candidates list is empty.
    """
    if not candidates:
        return None

    best: Stage4Result | None = None

    q_domain  = (query_metadata.get('metadata') or {}).get('domain', '')
    q_len     = float(query_metadata.get('context_len') or 512)
    q_norm    = float(query_metadata.get('l2_norm') or 1.0)
    q_desc    = (query_metadata.get('task_description') or '')[:80]

    for cand in candidates:
        c_domain  = (cand.get('metadata') or {}).get('domain', '')
        c_len     = float(cand.get('context_len') or 512)
        c_norm    = float(cand.get('l2_norm') or 1.0)
        c_desc    = (cand.get('task_description') or '')[:80]
        c_id      = cand.get('cache_id', '')

        domain_match  = (q_domain and c_domain and q_domain == c_domain)
        domain_score  = domain_bonus if domain_match else 0.0
        len_score     = 1.0 - min(abs(q_len - c_len) / max(max_len, 1), 1.0)
        norm_score    = 1.0 - min(abs(q_norm - c_norm) / max(max_norm, 1), 1.0)
        kw_score      = _keyword_overlap(q_desc, c_desc)
        meta_score    = domain_score + len_score + norm_score + kw_score

        r = Stage4Result(
            doc_id         = c_id,
            meta_score     = meta_score,
            confidence     = 'low-metadata',
            metadata_used  = True,
            domain_matched = domain_match,
            score_breakdown = {
                'domain': round(domain_score, 3),
                'len':    round(len_score, 3),
                'norm':   round(norm_score, 3),
                'kw':     round(kw_score, 3),
            },
        )
        if best is None or meta_score > best.meta_score:
            best = r

    return best

"""
ENGRAM constrained retrieval — apophatic negative constraint layer.

Implements constrained_retrieve() which penalizes candidates too
similar to known confusion partners, resolving dense-region failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class CosineResult:
    """Single retrieval result."""

    doc_id: str
    score: float
    cos_score: float
    margin: float = 0.0
    constrained: bool = False


@dataclass
class EngramQuery:
    """Query with optional negative constraints.

    like:        Fingerprint to match (positive constraint).
    unlike:      Fingerprints to avoid (negative constraints).
    min_margin:  Minimum acceptable score gap.
    fingerprint: Which fingerprint field to use ('fourier', 'fcdb', 'perdoc').
    """

    like: torch.Tensor
    unlike: list[torch.Tensor] = field(default_factory=list)
    min_margin: float = 0.001
    domain_hint: str | None = None
    fingerprint: str = "fourier"


def cosine_search(
    query_fp: torch.Tensor,
    index: dict[str, torch.Tensor],
    top_k: int = 5,
) -> list[CosineResult]:
    """Standard unconstrained cosine similarity search."""
    if not index:
        return []

    doc_ids = list(index.keys())
    matrix = torch.stack([index[d] for d in doc_ids])
    qn = F.normalize(query_fp.unsqueeze(0).float(), dim=-1)
    mn = F.normalize(matrix.float(), dim=-1)
    sims = (qn @ mn.T).squeeze(0)

    top_indices = sims.topk(min(top_k, len(doc_ids))).indices.tolist()
    results = [
        CosineResult(
            doc_id=doc_ids[i],
            score=float(sims[i].item()),
            cos_score=float(sims[i].item()),
        )
        for i in top_indices
    ]
    if len(results) >= 2:
        results[0].margin = results[0].score - results[1].score
    return results


def constrained_retrieve(
    query: EngramQuery,
    index: dict[str, torch.Tensor],
    top_k: int = 5,
    neg_weight: float = 0.5,
    neg_threshold: float = 0.85,
) -> list[CosineResult]:
    """Retrieval with negative (apophatic) constraint layer.

    Penalizes candidates too similar to `unlike` fingerprints.
    In dense regions, this discriminates between docs that would
    otherwise have identical cosine scores.

    Algorithm:
        1. Compute cosine similarity to query (positive score)
        2. For each unlike fingerprint, compute sim to each candidate
        3. Subtract penalty: neg_weight * max(0, sim_to_unlike - threshold)
        4. Sort by adjusted score
    """
    if not index:
        return []

    doc_ids = list(index.keys())
    matrix = torch.stack([index[d] for d in doc_ids])
    qn = F.normalize(query.like.unsqueeze(0).float(), dim=-1)
    mn = F.normalize(matrix.float(), dim=-1)
    cos_scores = (qn @ mn.T).squeeze(0)

    adjusted = cos_scores.clone()

    if query.unlike:
        for unlike_fp in query.unlike:
            un = F.normalize(unlike_fp.unsqueeze(0).float(), dim=-1)
            neg_sims = (un @ mn.T).squeeze(0)
            penalty = neg_weight * torch.clamp(neg_sims - neg_threshold, min=0)
            adjusted = adjusted - penalty

    top_indices = adjusted.topk(min(top_k, len(doc_ids))).indices.tolist()
    results = [
        CosineResult(
            doc_id=doc_ids[i],
            score=float(adjusted[i].item()),
            cos_score=float(cos_scores[i].item()),
            constrained=bool(query.unlike),
        )
        for i in top_indices
    ]
    if len(results) >= 2:
        results[0].margin = results[0].score - results[1].score
    return results


# ── TWO-STAGE GEODESIC RETRIEVAL ──────────────────────────────────────

from enum import Enum


class RetrievalConfidence(Enum):
    HIGH = "high"  # margin > 5x threshold, single pass sufficient
    MEDIUM = "medium"  # margin > threshold, or resolved by stage-2
    LOW = "low"  # margin < threshold after stage-2 — uncertain


@dataclass
class GeodesicResult:
    """Result from geodesic_retrieve()."""

    doc_id: str
    score: float
    margin: float
    confidence: RetrievalConfidence
    stages_used: int = 1  # 1 = single pass, 2 = two-stage, 3 = constrained
    constraint_used:  bool = False
    stage4_used:      bool = False
    stage4_doc_id:    str  = ""


def geodesic_retrieve(
    query_fp: torch.Tensor,
    hnsw_index,  # EngramIndex instance
    eng_index: dict,  # {doc_id: eng_data} for constraint layer
    margin_threshold: float = 0.005,
    correction_weight: float = 0.3,
    top_k: int = 5,
) -> GeodesicResult:
    """
    Two-stage geodesic retrieval with automatic confidence scoring.

    Stage 1: HNSW approximate nearest-neighbor search.
             If margin(top1, top2) >= margin_threshold -> HIGH or MEDIUM.
             Return immediately.

    Stage 2: Activated when margin < margin_threshold.
             Interpolate query fingerprint toward Stage-1 top-1 result.
             The interpolation weight (correction_weight=0.3) bends the
             geodesic toward the probable destination without assuming
             the Stage-1 answer is correct.
             If Stage-2 margin >= threshold -> MEDIUM confidence.
             If Stage-2 margin still < threshold -> LOW confidence.

    Stage 3: If confusion_flag is set on Stage-2 top result AND
             eng_index is provided -> activate negative constraint.
             Uses the confusion partner fingerprint as unlike constraint.

    Args:
        query_fp:          [dim] query fingerprint (v2 recommended).
        hnsw_index:        Built EngramIndex instance.
        eng_index:         Dict {doc_id: eng_data} loaded from .eng files.
                           Used for Stage-2 interpolation and Stage-3
                           constraint layer. Pass empty dict {} to disable.
        margin_threshold:  Minimum margin for MEDIUM confidence.
                           Default 0.005 (below S3 mean margin of 0.009).
        correction_weight: Interpolation weight for Stage-2 trajectory
                           correction. 0.3 = 30% pull toward top-1.
                           Range: 0.1 (gentle) to 0.5 (aggressive).
        top_k:             Candidates per search pass.

    Returns:
        GeodesicResult with doc_id, score, margin, confidence, stages_used.

    Usage:
        result = geodesic_retrieve(query_fp, idx, eng_index={})
        if result.confidence == RetrievalConfidence.LOW:
            # Flag for human review or return with uncertainty warning
            pass
    """
    # Stage 1: HNSW search
    s1_results = hnsw_index.search(query_fp, top_k=top_k)
    if len(s1_results) < 2:
        return GeodesicResult(
            doc_id=s1_results[0].doc_id if s1_results else "",
            score=s1_results[0].score if s1_results else 0.0,
            margin=0.0,
            confidence=RetrievalConfidence.LOW,
            stages_used=1,
        )

    s1_margin = s1_results[0].margin

    # High confidence: single pass sufficient
    if s1_margin >= margin_threshold * 5:
        return GeodesicResult(
            doc_id=s1_results[0].doc_id,
            score=s1_results[0].score,
            margin=s1_margin,
            confidence=RetrievalConfidence.HIGH,
            stages_used=1,
        )

    # Medium confidence: above threshold but not high
    if s1_margin >= margin_threshold:
        return GeodesicResult(
            doc_id=s1_results[0].doc_id,
            score=s1_results[0].score,
            margin=s1_margin,
            confidence=RetrievalConfidence.MEDIUM,
            stages_used=1,
        )

    # Stage 2: trajectory correction
    # Retrieve top-1 fingerprint from eng_index for interpolation
    top1_id = s1_results[0].doc_id
    top1_eng = eng_index.get(top1_id, {})
    top1_fp = top1_eng.get("vec_fourier_v2")
    if top1_fp is None:
        top1_fp = top1_eng.get("vec_fourier")

    if top1_fp is not None:
        # Bend geodesic toward Stage-1 top-1
        refined_fp = F.normalize(
            (1 - correction_weight) * query_fp.float()
            + correction_weight * top1_fp.float(),
            dim=-1,
        )
        s2_results = hnsw_index.search(refined_fp, top_k=top_k)
        s2_margin = s2_results[0].margin if len(s2_results) >= 2 else 0.0

        # Stage 3: check confusion_flag on Stage-2 top result
        s2_top_id = s2_results[0].doc_id if s2_results else top1_id
        s2_top_eng = eng_index.get(s2_top_id, {})

        if s2_top_eng.get("confusion_flag") and eng_index:
            # Activate negative constraint: find confusion partner fps
            def _pick_fp(d: dict) -> torch.Tensor | None:
                v = d.get("vec_fourier_v2")
                return v if v is not None else d.get("vec_fourier")

            confusion_fps = [
                _pick_fp(d)
                for did, d in eng_index.items()
                if d.get("confusion_flag")
                and did != s2_top_id
                and _pick_fp(d) is not None
            ]
            if confusion_fps:
                # Build flat index for constrained_retrieve
                flat_index = {
                    did: _pick_fp(d)
                    for did, d in eng_index.items()
                    if _pick_fp(d) is not None
                }
                q_constrained = EngramQuery(
                    like=refined_fp,
                    unlike=confusion_fps[:3],  # top 3 confusion partners
                    min_margin=margin_threshold,
                )
                s3_results = constrained_retrieve(
                    q_constrained,
                    flat_index,
                )
                if s3_results:
                    s3_margin = s3_results[0].margin
                    s3_conf = (
                        RetrievalConfidence.MEDIUM
                        if s3_margin >= margin_threshold
                        else RetrievalConfidence.LOW
                    )
                    return GeodesicResult(
                        doc_id=s3_results[0].doc_id,
                        score=s3_results[0].score,
                        margin=s3_margin,
                        confidence=s3_conf,
                        stages_used=3,
                        constraint_used=True,
                    )

        if s2_margin >= margin_threshold:
            return GeodesicResult(
                doc_id=s2_top_id,
                score=s2_results[0].score,
                margin=s2_margin,
                confidence=RetrievalConfidence.MEDIUM,
                stages_used=2,
            )
        else:
            # Both stages low margin — return LOW confidence
            return GeodesicResult(
                doc_id=s2_top_id,
                score=s2_results[0].score,
                margin=s2_margin,
                confidence=RetrievalConfidence.LOW,
                stages_used=2,
            )
    else:
        # No vector for interpolation — return Stage-1 with LOW confidence
        return GeodesicResult(
            doc_id=top1_id,
            score=s1_results[0].score,
            margin=s1_margin,
            confidence=RetrievalConfidence.LOW,
            stages_used=1,
        )



def geodesic_retrieve_with_prior(
    query_fp: torch.Tensor,
    hnsw_index,
    eng_index: dict,
    index_c=None,
    query_doc_id: str | None = None,
    margin_threshold: float = 0.005,
    correction_weight: float = 0.3,
    top_k: int = 5,
) -> GeodesicResult:
    """
    Prior-aware geodesic retrieval. Uses IndexC history to pre-apply
    constraints on known chronic failures — skipping Stages 1 and 2.

    When index_c and query_doc_id are provided:
      - If doc is a chronic failure: apply Stage 3 (constraint) immediately.
        This avoids 2 wasted HNSW passes before getting to the constraint.
      - If doc has prior LOW history (not yet chronic): lower threshold.
      - If no prior: standard 3-stage geodesic_retrieve().

    Args:
        query_fp:         [dim] query fingerprint.
        hnsw_index:       Built EngramIndex instance.
        eng_index:        {doc_id: eng_data} from .eng files.
        index_c:          IndexC instance, or None to disable prior mode.
        query_doc_id:     doc_id being queried (for prior lookup).
        margin_threshold: Base margin threshold. Lowered if prior is LOW.
        correction_weight: Stage-2 interpolation weight.
        top_k:            Candidates per HNSW pass.

    Returns:
        GeodesicResult. For chronic failures: stages_used=0 (preempted).
    """
    if index_c is None or query_doc_id is None:
        return geodesic_retrieve(
            query_fp, hnsw_index, eng_index,
            margin_threshold=margin_threshold,
            correction_weight=correction_weight,
            top_k=top_k,
        )

    prior = index_c.prior(query_doc_id)

    # Preemptive mode: known chronic failure
    if prior.is_chronic_failure and prior.n_total >= 2:
        pairs = index_c.confusion_registry(min_confusions=1)
        partners = [
            p.doc_b for p in pairs if p.doc_a == query_doc_id
        ] + [
            p.doc_a for p in pairs if p.doc_b == query_doc_id
        ]

        unlike_fps = []
        for partner_id in partners[:3]:
            partner_eng = eng_index.get(partner_id, {})
            fp = partner_eng.get("vec_fourier_v2")
            if fp is None:
                fp = partner_eng.get("vec_fourier")
            if fp is not None:
                unlike_fps.append(fp)

        if unlike_fps:
            flat_index: dict[str, torch.Tensor] = {}
            for did, d in eng_index.items():
                v = d.get("vec_fourier_v2")
                if v is None:
                    v = d.get("vec_fourier")
                if v is not None:
                    flat_index[did] = v

            q_constrained = EngramQuery(
                like=query_fp,
                unlike=unlike_fps,
                min_margin=margin_threshold,
            )
            s3 = constrained_retrieve(q_constrained, flat_index)
            if s3:
                s3_margin = s3[0].margin
                return GeodesicResult(
                    doc_id=s3[0].doc_id,
                    score=s3[0].score,
                    margin=s3_margin,
                    confidence=(
                        RetrievalConfidence.MEDIUM
                        if s3_margin >= margin_threshold
                        else RetrievalConfidence.LOW
                    ),
                    stages_used=0,
                    constraint_used=True,
                )

    # Prior LOW but not yet chronic: tighten threshold
    if prior.n_low > 0 and prior.n_total > 0:
        low_frac = prior.n_low / prior.n_total
        margin_threshold = margin_threshold * (1 + low_frac)

    return geodesic_retrieve(
        query_fp, hnsw_index, eng_index,
        margin_threshold=margin_threshold,
        correction_weight=correction_weight,
        top_k=top_k,
    )


def geodesic_retrieve_stage4(
    query_fp: torch.Tensor,
    hnsw_index,
    eng_index: dict,
    query_metadata: dict | None = None,
    index_c=None,
    query_doc_id: str | None = None,
    margin_threshold: float = 0.005,
    correction_weight: float = 0.3,
    top_k: int = 5,
) -> GeodesicResult:
    """
    Full pipeline: prior-aware geodesic retrieval with Stage 4 fallback.

    Extends geodesic_retrieve_with_prior() with a Stage 4 metadata
    disambiguation layer. When confidence is LOW and query_metadata
    is provided, calls metadata_disambiguate() on top candidates
    from the last HNSW pass before giving up.

    Confidence tier:
      HIGH         -> fingerprint, 0% error rate target
      MEDIUM       -> fingerprint, 0% error rate target
      LOW          -> fingerprint failed, Stage 4 unavailable
      low-metadata -> fingerprint failed, Stage 4 used secondary signal

    Args:
        query_fp:       [dim] query fingerprint (vec_fourier_v2).
        hnsw_index:     Built EngramIndex.
        eng_index:      {doc_id: eng_data} from .eng files.
        query_metadata: dict with task_description, context_len, l2_norm,
                        metadata.domain -- from the query doc's .eng data.
                        If None, Stage 4 is disabled.
        index_c:        IndexC instance for prior lookup.
        query_doc_id:   doc_id of the query source (for priors).
        margin_threshold, correction_weight, top_k: as in base function.
    """
    from kvcos.engram.metadata_disambiguate import metadata_disambiguate

    base = geodesic_retrieve_with_prior(
        query_fp, hnsw_index, eng_index,
        index_c=index_c,
        query_doc_id=query_doc_id,
        margin_threshold=margin_threshold,
        correction_weight=correction_weight,
        top_k=top_k,
    )

    # Only activate Stage 4 on LOW confidence with metadata available
    if base.confidence != RetrievalConfidence.LOW:
        return base
    if query_metadata is None:
        return base

    # Get top candidates from HNSW for metadata scoring
    candidates_hnsw = hnsw_index.search(query_fp, top_k=5)
    candidates_meta = [
        eng_index[r.doc_id]
        for r in candidates_hnsw
        if r.doc_id in eng_index
    ]

    if not candidates_meta:
        return base

    s4 = metadata_disambiguate(candidates_meta, query_metadata)
    if s4 is None:
        return base

    return GeodesicResult(
        doc_id           = s4.doc_id,
        score            = base.score,
        margin           = base.margin,
        confidence       = RetrievalConfidence.LOW,   # still LOW
        stages_used      = base.stages_used,
        constraint_used  = base.constraint_used,
        stage4_used      = True,
        stage4_doc_id    = s4.doc_id,
    )
    # Note: confidence stays LOW -- Stage 4 is a tiebreaker, not a promotion.
    # Callers should check stage4_used=True to distinguish "failed silently"
    # from "failed with secondary signal". The confidence tier string
    # 'low-metadata' is available via Stage4Result for logging.

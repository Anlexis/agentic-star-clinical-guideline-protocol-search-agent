"""AgentCore Platform v1.0"""

# HCR-C2-005 - RerankFilterNode
# Domain node 3: rerank the retrieval candidates and enforce the life-safety
# relevance floor. Deterministic: a small category-match boost on top of the
# retrieval score, drop everything below `score_threshold` (0.75 - the
# healthcare floor), cap the survivors at `top_k`. This is the mechanism
# behind the domain's "decline when retrieval confidence is insufficient"
# requirement: GenerateAnswerNode only ever sees passages that cleared this
# floor.
#
# execute(self, state) -> dict only - NO config parameter. Retrieval tuning
# (top_k / score_threshold) is read from the SCALAR state fields
# retrieval_top_k / retrieval_score_threshold, republished by
# DomainWorkflowGraph._extra_initial_state() from the `retrieval` block. Falls
# back to module defaults (mirroring config/config.yaml) when those state
# fields are unseeded (unit tests). Caller-supplied top_k / score_threshold
# overrides (query_filters, validated by InputValidateNode) win when they are
# STRICTER - a caller may narrow the result set or raise the floor, never
# widen or lower it.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.schemas.agent_status import AgentStatus
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json

# Module defaults - mirror the `retrieval` block in config/config.yaml
# (life-safety relevance floor: score_threshold 0.75).
_DEFAULT_TOP_K = 8
_DEFAULT_SCORE_THRESHOLD = 0.75

# Boost applied when a candidate's category matches the caller's filter.
_CATEGORY_BOOST = 0.1


def _resolve_top_k(state: AgentState) -> int:
    top_k = state.get("retrieval_top_k")
    if isinstance(top_k, bool):
        return _DEFAULT_TOP_K
    if isinstance(top_k, int):
        return top_k
    return _DEFAULT_TOP_K


def _resolve_score_threshold(state: AgentState) -> float:
    threshold = state.get("retrieval_score_threshold")
    if isinstance(threshold, bool):
        return _DEFAULT_SCORE_THRESHOLD
    if isinstance(threshold, (int, float)):
        return float(threshold)
    return _DEFAULT_SCORE_THRESHOLD


class RerankFilterNode(FunctionNode):
    """Rerank candidates, apply the relevance floor, cap at top_k.

    Input state keys:
        retrieved_documents:        JSON list of scored candidates (from RetrieveNode)
        query_filters:               JSON dict with optional category / top_k /
                                      score_threshold override
        retrieval_top_k:             state-seeded runtime value (scalar)
        retrieval_score_threshold:   state-seeded runtime value (scalar)

    Output state keys (partial dict):
        ranked_documents: JSON list of surviving passages (score desc, <= top_k)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        candidates: List[Dict[str, Any]] = from_json(state.get("retrieved_documents"), []) or []
        filters = from_json(state.get("query_filters"), {}) or {}

        top_k = max(1, min(20, _resolve_top_k(state)))
        # A stricter caller override (validated by InputValidateNode) wins.
        caller_top_k = filters.get("top_k")
        if isinstance(caller_top_k, int) and 1 <= caller_top_k < top_k:
            top_k = caller_top_k

        score_threshold = max(0.0, min(1.0, _resolve_score_threshold(state)))
        # A stricter caller override (validated by InputValidateNode, which
        # rejects any value below the configured floor) wins.
        caller_threshold = filters.get("score_threshold")
        if (
            isinstance(caller_threshold, (int, float))
            and not isinstance(caller_threshold, bool)
            and score_threshold < float(caller_threshold) <= 1.0
        ):
            score_threshold = float(caller_threshold)

        category = filters.get("category")

        reranked: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            entry = dict(candidate)  # local copy - inputs stay immutable
            try:
                score = float(entry.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            if category and str(entry.get("category", "")).lower() == str(category).lower():
                score = min(1.0, score + _CATEGORY_BOOST)
            entry["score"] = round(score, 4)
            reranked.append(entry)

        # Deterministic ordering: score desc, then id asc for stable ties.
        reranked.sort(key=lambda c: (-c.get("score", 0.0), str(c.get("id", ""))))

        kept = [c for c in reranked if c.get("score", 0.0) >= score_threshold][:top_k]
        dropped = len(reranked) - len(kept)

        # Domain audit: rerank + relevance floor applied.
        emit_trace_event(
            "rerank_filter_complete",
            {
                "kept": len(kept),
                "dropped": dropped,
                "score_threshold": score_threshold,
                "top_k": top_k,
            },
            state,
        )

        return {"ranked_documents": to_json(kept)}

"""AgentCore Platform v1.0"""

# State must be a flat TypedDict - never a Pydantic BaseModel. Graph
# checkpoints use msgpack serialization; Pydantic objects cause silent
# corruption.  Extend AgentState with agent-specific fields only.  Do NOT add
# credentials, secrets, or Pydantic models.
#
# Msgpack safety: structured fields (dict / list[dict]) are stored as JSON
# STRINGS, not bare Python containers - a bare dict/list in a checkpointed
# State field is a state-safety violation. Producers serialize with to_json()
# on write; consumers deserialize with from_json() on read.
#
# HCR-C2-005 - Clinical Guidelines & Protocol KB Search Agent (Cat 2 RAG).
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Fields below cover both layers.
#
# Life-safety domain note: this template answers strictly from the seeded
# clinical-guideline KB and declines (does not answer from parametric
# knowledge) when retrieval confidence is insufficient - see GenerateAnswerNode.
# No patient-record / PHI ingestion: only the normalised clinical question,
# guideline passage summaries, and the final grounded answer are persisted.
#
# Domain nodes take NO config parameter in execute(). Retrieval tuning
# (config/config.yaml `retrieval` block) is forwarded by
# ClinicalGuidelineSearchGraphNode._parent_config() to the inner graph's
# constructor, then republished into these SCALAR state fields by
# DomainWorkflowGraph._extra_initial_state() - RetrieveNode / RerankFilterNode
# read them straight off state, falling back to module defaults when unseeded
# (e.g. unit tests that construct a node directly).

import json
from typing import Any, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (msgpack safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for HCR-C2-005.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    Domain fields are Optional (default-absent) so the state is valid at
    graph initialisation, before any node has written a value.
    """

    # ------------------------------------------------------------------
    # Outer layer - set by PreProcessNode / ClinicalGuidelineSearchGraphNode.merge_output
    # ------------------------------------------------------------------

    # Identifier-stripped, validated clinical-question payload produced by
    # PreProcessNode. Raw input is NOT persisted beyond PreProcessNode.
    validated_input: Optional[str]

    # Final clinical-guideline KB search answer, mapped from the inner
    # graph's formatted_answer output via merge_output.
    guideline_answer: Optional[str]

    # ------------------------------------------------------------------
    # Inner layer - domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # InputValidateNode outputs
    # Normalised free-text clinical question (whitespace-collapsed, length-capped).
    search_query: Optional[str]

    # JSON STRING (to_json) of the validated structured query params.
    # Deserialised dict shape: {"category": str | None, "top_k": int | None,
    # "score_threshold": float | None}.
    # Consumers (RetrieveNode, RerankFilterNode) read it back via from_json().
    query_filters: Optional[str]

    # Runtime `retrieval` block, republished as SCALAR state fields by
    # DomainWorkflowGraph._extra_initial_state() (nodes take no config
    # parameter - config flows through State, not through execute()).
    # Consumers (RetrieveNode, RerankFilterNode) fall back to module defaults
    # mirroring config/config.yaml when these are unseeded.
    retrieval_top_k: Optional[int]
    retrieval_score_threshold: Optional[float]
    retrieval_kb_path: Optional[str]

    # RetrieveNode output
    # JSON STRING (to_json) of scored KB candidates. Deserialised shape:
    # list[dict], each entry {"id": str, "title": str, "category": str,
    # "source": str, "score": float, "excerpt": str}.
    # Consumers (RerankFilterNode) read it back via from_json().
    retrieved_documents: Optional[str]

    # RerankFilterNode output
    # JSON STRING (to_json) of reranked + threshold-filtered passages, capped
    # at top_k. Same entry shape as retrieved_documents.
    # Consumers (GenerateAnswerNode) read it back via from_json().
    ranked_documents: Optional[str]

    # GenerateAnswerNode outputs
    # Rule-assembled grounded answer body with numbered citation markers, or
    # the explicit insufficient-evidence decline when no passage clears the
    # relevance threshold.
    grounded_answer: Optional[str]

    # JSON STRING (to_json) of citations. Deserialised shape: list[dict],
    # each entry {"ref": int, "id": str, "title": str, "source": str}.
    # Consumers (OutputFormatNode) read it back via from_json().
    citations: Optional[str]

    # OutputFormatNode output
    # Final formatted answer (body + sources + the mandatory advisory
    # disclaimer). Written by OutputFormatNode; surfaced to the outer graph
    # via get_output() -> merge_output().
    formatted_answer: Optional[str]

    # Validation / parse notes accumulated during intake (no patient data).
    # JSON STRING (to_json) of list[str].
    intake_notes: Optional[str]

    # ------------------------------------------------------------------
    # Tracing / audit - framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    error_code: Optional[str]
    # node_history inherited from AgentState; listed here for clarity
    # node_history: Optional[List[str]]

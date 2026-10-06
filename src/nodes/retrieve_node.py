"""AgentCore Platform v1.0"""

# HCR-C2-005 - RetrieveNode
# Domain node 2: deterministic keyword retrieval over the seeded clinical
# guideline knowledge base (config/kb/hcr_clinical_guideline_kb.json). v1 is
# fully deterministic - no embedding model or vector store; the retrieval
# contract (retrieved_documents JSON) is store-agnostic so a later
# vector-store upgrade only swaps this node's internals.
#
# execute(self, state) -> dict only - NO config parameter. Retrieval tuning
# (top_k / kb_path) is read from the SCALAR state fields retrieval_top_k /
# retrieval_kb_path, republished by DomainWorkflowGraph._extra_initial_state()
# from the `retrieval` block forwarded via
# ClinicalGuidelineSearchGraphNode._parent_config(). Falls back to module
# defaults (mirroring config/config.yaml) when those state fields are unseeded
# - e.g. a unit test constructing this node directly and calling execute()
# with a minimal state.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
import re
from pathlib import Path
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
_DEFAULT_KB_PATH = "config/kb/hcr_clinical_guideline_kb.json"

# Repo root: src/nodes/retrieve_node.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Minimal stopword set for query tokenisation (deterministic, no NLP deps).
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "with",
        "under",
        "what",
        "which",
        "when",
        "how",
        "do",
        "does",
        "must",
        "should",
        "before",
        "after",
        "by",
        "at",
        "from",
        "that",
        "this",
        "it",
        "as",
        "was",
        "were",
        "can",
        "may",
        "any",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Per-field match weights: a query token found in the title counts more than
# one found only in the body content.
_TITLE_WEIGHT = 1.0
_TAG_WEIGHT = 0.8
_CONTENT_WEIGHT = 0.5

# Excerpt length carried into retrieved_documents (keeps State small).
_EXCERPT_CHARS = 400


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, stopwords and 1-2 char noise removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2 and t not in _STOPWORDS]


def _resolve_top_k(state: AgentState) -> int:
    """Effective top_k: state-seeded retrieval_top_k > module default."""
    top_k = state.get("retrieval_top_k")
    if isinstance(top_k, bool):
        return _DEFAULT_TOP_K
    if isinstance(top_k, int):
        return top_k
    return _DEFAULT_TOP_K


def _resolve_kb_path(state: AgentState) -> str:
    """Effective kb_path: state-seeded retrieval_kb_path > module default."""
    kb_path = state.get("retrieval_kb_path")
    if isinstance(kb_path, str) and kb_path:
        return kb_path
    return _DEFAULT_KB_PATH


def _load_kb(kb_path: str) -> tuple[List[Dict[str, Any]], List[str]]:
    """Load the seeded KB JSON. Missing / malformed file degrades gracefully."""
    notes: List[str] = []
    path = Path(kb_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        notes.append(f"RetrieveNode: knowledge base not readable at {kb_path}.")
        return [], notes
    if not isinstance(entries, list):
        notes.append("RetrieveNode: knowledge base root must be a JSON list.")
        return [], notes
    return [e for e in entries if isinstance(e, dict)], notes


def _score_entry(entry: Dict[str, Any], query_tokens: List[str]) -> float:
    """Per-entry relevance: best field-weight per query token, averaged."""
    if not query_tokens:
        return 0.0
    title_tokens = set(_tokenize(str(entry.get("title", ""))))
    tag_tokens = set(_tokenize(" ".join(str(t) for t in entry.get("tags", []))))
    content_tokens = set(_tokenize(str(entry.get("content", ""))))
    total = 0.0
    for token in query_tokens:
        if token in title_tokens:
            total += _TITLE_WEIGHT
        elif token in tag_tokens:
            total += _TAG_WEIGHT
        elif token in content_tokens:
            total += _CONTENT_WEIGHT
    return round(total / len(query_tokens), 4)


class RetrieveNode(FunctionNode):
    """Score the seeded clinical-guideline KB against the search query.

    Input state keys:
        search_query:      normalised question (from InputValidateNode)
        query_filters:      JSON dict with optional category filter
        retrieval_top_k:    state-seeded runtime value (scalar)
        retrieval_kb_path:  state-seeded runtime value (scalar)

    Output state keys (partial dict):
        retrieved_documents: JSON list of scored candidates (score desc)
        intake_notes:        (on KB anomalies) JSON list[str]
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        query = state.get("search_query") or state.get("validated_input") or state.get("user_input", "")
        filters = from_json(state.get("query_filters"), {}) or {}

        top_k = max(1, min(20, _resolve_top_k(state)))
        kb_path = _resolve_kb_path(state)

        entries, notes = _load_kb(kb_path)

        category = filters.get("category")
        if category:
            entries = [e for e in entries if str(e.get("category", "")).lower() == str(category).lower()]

        query_tokens = _tokenize(query if isinstance(query, str) else "")

        candidates: List[Dict[str, Any]] = []
        for entry in entries:
            score = _score_entry(entry, query_tokens)
            if score <= 0.0:
                continue
            candidates.append(
                {
                    "id": str(entry.get("id", "")),
                    "title": str(entry.get("title", "")),
                    "category": str(entry.get("category", "")),
                    "source": str(entry.get("source", "")),
                    "score": score,
                    "excerpt": str(entry.get("content", ""))[:_EXCERPT_CHARS],
                }
            )

        # Deterministic ordering: score desc, then id asc for stable ties.
        candidates.sort(key=lambda c: (-c["score"], c["id"]))
        # Keep a candidate pool wider than top_k - RerankFilterNode makes
        # the final cut after the category boost + relevance floor.
        pool_size = max(top_k * 3, 10)
        candidates = candidates[:pool_size]

        # Domain audit: retrieval pass completed.
        emit_trace_event(
            "retrieve_complete",
            {
                "candidates": len(candidates),
                "kb_entries": len(entries),
                "query_tokens": len(query_tokens),
                "top_k": top_k,
            },
            state,
        )

        out: Dict[str, Any] = {"retrieved_documents": to_json(candidates)}
        if notes:
            # Append to (never clobber) the notes accumulated upstream.
            prior = from_json(state.get("intake_notes"), []) or []
            out["intake_notes"] = to_json(list(prior) + notes)
        return out

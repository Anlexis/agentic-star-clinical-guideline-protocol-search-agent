# HCR-C2-005 — Unit Tests: RetrieveNode (inner domain node 2)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller
# — EVERY call, including the config-knob tests. C2 (the old direct
# execute(state, config=...) carve-out) is RETIRED 2026-07-27: RetrieveNode
# takes no config parameter at all. Retrieval tuning (top_k / kb_path) is read
# from the SCALAR state fields retrieval_top_k / retrieval_kb_path (seeded by
# DomainWorkflowGraph._extra_initial_state() in production; seeded directly on
# the state dict here to exercise the knob) — never through execute()'s
# signature.
#
# Mirrors docs/03_test_spec.md §2.3 (RET-01..RET-08).
# Deterministic — keyword scoring over the seeded
# config/kb/hcr_clinical_guideline_kb.json; no LLM, no network.
# framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json

# Same text as deploy/invoke_payload.json's "input" (PB-6's _VALID_PAYLOAD) —
# lowercase, PII-free clinical phrasing (C3: no names/identifiers/digit groups).
_CAP_QUERY = (
    "what is the current first-line empiric antibiotic for community-acquired "
    "pneumonia under our institutional stewardship protocol?"
)


def _make_state(query=_CAP_QUERY, **extra) -> dict:
    state = {
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestRetrieveHappyPath:
    def test_ret_01_top_hit_is_cap_entry(self):
        result = RetrieveNode()(_make_state())
        docs = from_json(result["retrieved_documents"])
        assert docs, "expected candidates for the CAP/antibiotic-stewardship query"
        assert docs[0]["id"] == "kb-001"

    def test_ret_02_scores_sorted_descending(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        scores = [d["score"] for d in docs]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0.0 for s in scores)

    def test_ret_03_entry_shape_and_excerpt_cap(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        for doc in docs:
            assert set(doc.keys()) == {"id", "title", "category", "source", "score", "excerpt"}
            assert len(doc["excerpt"]) <= 400

    def test_retrieved_documents_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = RetrieveNode()(_make_state())
        assert isinstance(result["retrieved_documents"], str)


class TestRetrieveFilters:
    def test_ret_04_category_filter_restricts_pool(self):
        # kb-002 (renal dose adjustment) and kb-009 (contrast nephropathy) are
        # the only two "renal_dosing" entries; "renal" is a title token of
        # kb-002 and only a content token of kb-009, so both score > 0 and the
        # category filter must restrict the pool to exactly these two.
        state = _make_state(
            query="renal",
            query_filters=to_json({"category": "renal_dosing", "top_k": None}),
        )
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs, "renal_dosing category has seeded entries"
        assert {d["category"] for d in docs} == {"renal_dosing"}
        assert {d["id"] for d in docs} == {"kb-002", "kb-009"}
        assert docs[0]["id"] == "kb-002"  # title hit outranks the content-only hit

    def test_ret_05_empty_query_yields_no_candidates(self):
        docs = from_json(RetrieveNode()(_make_state(query=""))["retrieved_documents"])
        assert docs == []


class TestRetrieveStateSeededConfig:
    """Config plumbing: retrieval_kb_path is read off STATE, never execute()'s
    signature (C2 retired — RetrieveNode takes no config parameter at all).
    Every call below goes through node(state) per C1."""

    def test_ret_06_state_seeded_kb_path_override(self):
        state = _make_state(retrieval_kb_path="config/kb/does_not_exist.json")
        result = RetrieveNode()(state)
        assert from_json(result["retrieved_documents"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("not readable" in n for n in notes)

    def test_ret_07_unseeded_kb_path_falls_back_to_module_default(self):
        # No retrieval_kb_path in state — falls back to _DEFAULT_KB_PATH, which
        # mirrors config/agent.yaml and resolves to the real seeded KB.
        result = RetrieveNode()(_make_state())
        assert from_json(
            result["retrieved_documents"]
        ), "unseeded retrieval_kb_path must still resolve the real seeded KB"


class TestRetrieveNotesAccumulation:
    def test_ret_08_notes_append_never_clobber(self):
        state = _make_state(
            intake_notes=to_json(["earlier note from input validation"]),
            retrieval_kb_path="config/kb/does_not_exist.json",
        )
        result = RetrieveNode()(state)
        notes = from_json(result["intake_notes"])
        assert notes[0] == "earlier note from input validation"
        assert len(notes) == 2

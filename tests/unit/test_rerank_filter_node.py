# HCR-C2-005 — Unit Tests: RerankFilterNode (inner domain node 3)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller
# — EVERY call, including the config-knob tests. C2 (the old direct
# execute(state, config=...) carve-out) is RETIRED 2026-07-27: RerankFilterNode
# takes no config parameter at all. Retrieval tuning (top_k / score_threshold)
# is read from the SCALAR state fields retrieval_top_k / retrieval_score_threshold
# (seeded by DomainWorkflowGraph._extra_initial_state() in production; seeded
# directly on the state dict here to exercise the knob).
#
# HCR life-safety floor: the module default score_threshold is 0.75 (not the
# generic Cat-2 default) — this IS the mechanism behind the mandatory
# insufficient-evidence decline (docs/02_design.md). Test fixtures use scores
# either side of 0.75, not an arbitrary low bar.
#
# Mirrors docs/03_test_spec.md §2.4 (RRF-01..RRF-08).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.rerank_filter_node import RerankFilterNode
from src.schemas.state import from_json, to_json


def _doc(doc_id, score, category="sepsis"):
    return {
        "id": doc_id,
        "title": f"entry {doc_id}",
        "category": category,
        "source": "seeded kb",
        "score": score,
        "excerpt": "excerpt text",
    }


def _make_state(candidates, **extra) -> dict:
    state = {
        "retrieved_documents": to_json(candidates),
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestThresholdAndCap:
    def test_rrf_01_default_threshold_drops_weak_candidates(self):
        # 0.5 is a real, non-trivial retrieval score — but still below the HCR
        # life-safety floor (0.75), so it must be declined, not guessed.
        result = RerankFilterNode()(_make_state([_doc("kb-a", 0.9), _doc("kb-b", 0.5)]))
        kept = from_json(result["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]  # 0.5 < default 0.75 floor

    def test_rrf_02_state_seeded_score_threshold_override(self):
        # Canon: config knobs flow through STATE, invoked via node(state) (C1).
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.6)],
            retrieval_score_threshold=0.5,
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a", "kb-b"]  # 0.6 clears the lowered 0.5 floor

    def test_rrf_03_state_seeded_top_k_override(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.85), _doc("kb-c", 0.8)],
            retrieval_top_k=1,
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_ranked_documents_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = RerankFilterNode()(_make_state([_doc("kb-a", 0.9)]))
        assert isinstance(result["ranked_documents"], str)


class TestCategoryBoost:
    def test_rrf_04_matching_category_is_boosted_and_reranked(self):
        # Both candidates clear the 0.75 floor pre-boost; the category match
        # boost re-orders them (0.76 + 0.1 -> 0.86, ahead of the unboosted 0.80).
        state = _make_state(
            [_doc("kb-a", 0.80, category="pain_management"), _doc("kb-b", 0.76, category="sepsis")],
            query_filters=to_json({"category": "sepsis", "top_k": None}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-b", "kb-a"]
        assert kept[0]["score"] == 0.86  # 0.76 + 0.1 category boost

    def test_rrf_05_boost_is_capped_at_one(self):
        state = _make_state(
            [_doc("kb-a", 0.95, category="sepsis")],
            query_filters=to_json({"category": "sepsis", "top_k": None}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert kept[0]["score"] == 1.0


class TestCallerTopK:
    def test_rrf_06_stricter_caller_top_k_wins(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.85), _doc("kb-c", 0.8)],
            query_filters=to_json({"category": None, "top_k": 1}),
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_rrf_06_looser_caller_top_k_does_not_widen(self):
        state = _make_state(
            [_doc("kb-a", 0.9), _doc("kb-b", 0.85), _doc("kb-c", 0.8)],
            query_filters=to_json({"category": None, "top_k": 10}),
            retrieval_top_k=2,
        )
        kept = from_json(RerankFilterNode()(state)["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a", "kb-b"]


class TestRobustness:
    def test_rrf_07_garbage_candidates_are_skipped_or_dropped(self):
        candidates = [
            "not-a-dict",
            {"id": "kb-bad", "title": "b", "category": "x", "source": "s", "score": "NaN?", "excerpt": "e"},
            _doc("kb-a", 0.9),
        ]
        kept = from_json(RerankFilterNode()(_make_state(candidates))["ranked_documents"])
        # The string entry is skipped; the uncoercible score becomes 0.0 and
        # falls below the relevance floor.
        assert [d["id"] for d in kept] == ["kb-a"]

    def test_rrf_08_deterministic_tie_break_by_id(self):
        kept = from_json(RerankFilterNode()(_make_state([_doc("kb-b", 0.8), _doc("kb-a", 0.8)]))["ranked_documents"])
        assert [d["id"] for d in kept] == ["kb-a", "kb-b"]

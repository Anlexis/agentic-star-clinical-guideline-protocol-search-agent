# HCR-C2-005 — Unit Tests: retrieval quality over the seeded clinical KB
#
# Golden-query suite: drives the REAL inner retrieval chain
# (InputValidateNode → RetrieveNode → RerankFilterNode) via node(state) /
# __call__ (ANONYMOUS inner nodes) against
# config/kb/hcr_clinical_guideline_kb.json and pins the expected top hit per
# domain query. The scorer is deterministic (keyword field-weights, stable
# tie-break), so exact top-1 assertions are safe and catch KB / scorer /
# threshold regressions.
#
# HCR life-safety floor: RerankFilterNode's default score_threshold is 0.75
# (not the generic Cat-2 default) — every golden query below is phrased to
# echo its target entry's title tokens closely enough to clear that floor
# with a clean score of 1.0 (queries are built only from title tokens plus
# stopwords, which the scorer discards before scoring).
#
# Mirrors docs/03_test_spec.md §2.9 (QUAL-01..QUAL-07).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json
import pathlib

import pytest

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_KB_IDS = {
    entry["id"]
    for entry in json.loads((_ROOT / "config" / "kb" / "hcr_clinical_guideline_kb.json").read_text(encoding="utf-8"))
}

_DEFAULT_SCORE_THRESHOLD = 0.75  # mirrors config/agent.yaml retrieval block (HCR floor)


def _search(payload: str) -> list[dict]:
    """Run the real inner retrieval chain and return the surviving passages."""
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "quality-session",
        "execution_time": {},
    }
    state.update(InputValidateNode()(state))
    state.update(RetrieveNode()(state))
    state.update(RerankFilterNode()(state))
    return from_json(state["ranked_documents"], [])


# (query, expected top-1 KB entry id) — verified against the deterministic
# scorer; each query is built only from stopwords + the target entry's exact
# title tokens, so it clears the 0.75 HCR floor with a clean score of 1.0
# (the CAP query is the exception — same text as deploy/invoke_payload.json).
_GOLDEN_QUERIES = [
    (
        "what is the current first-line empiric antibiotic for community-acquired "
        "pneumonia under our institutional stewardship protocol?",
        "kb-001",
    ),
    ("what is the renal dose adjustment for intravenous vancomycin?", "kb-002"),
    ("what are the first-hour sepsis bundle actions for suspected septic shock?", "kb-004"),
    ("what is the penicillin allergy history and cephalosporin cross-reactivity?", "kb-006"),
    ("what is the acute ischemic stroke thrombolysis eligibility window?", "kb-007"),
    ("what is the pediatric weight-based dosing for acetaminophen?", "kb-010"),
]

# Zero token overlap with any seeded KB entry (matches test_domain_workflow_graph.py
# / test_graph_composition.py). Lowercase, PII-free (C3).
_NO_COVERAGE_QUERY = "what is the recommended elevator inspection interval for hospital facilities management"


class TestGoldenQueries:
    @pytest.mark.parametrize(("query", "expected_id"), _GOLDEN_QUERIES)
    def test_qual_01_top_hit_per_golden_query(self, query, expected_id):
        kept = _search(query)
        assert kept, f"no passage cleared the HCR relevance floor for: {query!r}"
        assert kept[0]["id"] == expected_id

    def test_qual_02_all_survivors_clear_the_relevance_floor(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["score"] >= _DEFAULT_SCORE_THRESHOLD

    def test_qual_03_survivor_ids_exist_in_the_seeded_kb(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["id"] in _KB_IDS


class TestPrecision:
    def test_qual_04_cap_query_keeps_only_the_infectious_disease_entry(self):
        # Off-topic passages score well below the HCR floor and are cut —
        # precision, not just recall.
        kept = _search(_GOLDEN_QUERIES[0][0])
        assert [d["id"] for d in kept] == ["kb-001"]

    def test_qual_05_category_filter_restricts_to_that_category(self):
        # kb-002 and kb-009 are the only "renal_dosing" entries. The category
        # filter restricts the candidate pool to both, but only kb-002 (a
        # title-level match) clears the 0.75 HCR floor — kb-009 (a
        # content-only match) does not. This is the relevance floor doing its
        # job WITHIN a category filter, not just across the whole KB.
        payload = json.dumps({"query": "renal", "category": "renal_dosing"})
        kept = _search(payload)
        assert kept, "renal_dosing category carries seeded entries"
        assert {d["category"] for d in kept} == {"renal_dosing"}
        assert kept[0]["id"] == "kb-002"


class TestNoCoverage:
    def test_qual_06_out_of_domain_query_yields_no_survivors(self):
        assert _search(_NO_COVERAGE_QUERY) == []

    def test_qual_07_no_coverage_produces_the_escalation_answer(self):
        state = {
            "ranked_documents": "[]",
            "search_query": _NO_COVERAGE_QUERY,
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
            "session_id": "quality-session",
            "execution_time": {},
        }
        result = GenerateAnswerNode()(state)
        assert "does not contain sufficient coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []

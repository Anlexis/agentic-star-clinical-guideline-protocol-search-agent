# HCR-C2-005 — Unit Tests: GenerateAnswerNode (inner domain node 4)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# The grounded answer / citations are DOMAIN fields (not framework mask
# targets), so Title-Case KB titles inside them are safe to assert on.
#
# Life-safety domain: this node is the mechanism behind BOTH required paths —
# the grounded-answer path (ranked_documents non-empty) and the mandatory
# insufficient-evidence decline (ranked_documents empty, because
# RerankFilterNode's 0.75 floor let nothing through) — see TestNoCoverage.
#
# Mirrors docs/03_test_spec.md §2.5 (GEN-01..GEN-05).
# Deterministic — rule-assembled from ranked_documents only (grounded by
# construction; no LLM, no network). framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.schemas.state import from_json, to_json


def _ranked(*entries):
    return to_json(list(entries))


def _doc(doc_id, title, excerpt, source="seeded kb"):
    return {
        "id": doc_id,
        "title": title,
        "category": "sepsis",
        "source": source,
        "score": 0.9,
        "excerpt": excerpt,
    }


def _make_state(ranked_documents, query="sepsis bundle first hour actions", **extra) -> dict:
    state = {
        "ranked_documents": ranked_documents,
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestGroundedAnswer:
    def test_gen_01_answer_carries_numbered_citation_markers(self):
        ranked = _ranked(
            _doc("kb-004", "Sepsis bundle first-hour actions", "obtain lactate, draw cultures before antibiotics."),
            _doc("kb-005", "Venous thromboembolism prophylaxis", "assess VTE risk on admission."),
        )
        result = GenerateAnswerNode()(_make_state(ranked))
        answer = result["grounded_answer"]
        assert "[1] Sepsis bundle first-hour actions:" in answer
        assert "[2] Venous thromboembolism prophylaxis:" in answer

    def test_gen_02_lead_sentence_quotes_the_query(self):
        ranked = _ranked(_doc("kb-004", "Sepsis bundle first-hour actions", "excerpt."))
        result = GenerateAnswerNode()(_make_state(ranked, query="sepsis bundle first hour actions"))
        assert 'the question: "sepsis bundle first hour actions"' in result["grounded_answer"]

    def test_gen_03_citations_mirror_ranked_order(self):
        ranked = _ranked(
            _doc("kb-004", "Sepsis bundle first-hour actions", "a.", source="Sepsis quality bundle"),
            _doc("kb-005", "Venous thromboembolism prophylaxis", "b."),
        )
        citations = from_json(GenerateAnswerNode()(_make_state(ranked))["citations"])
        assert [c["ref"] for c in citations] == [1, 2]
        assert [c["id"] for c in citations] == ["kb-004", "kb-005"]
        assert citations[0]["source"] == "Sepsis quality bundle"

    def test_citations_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        ranked = _ranked(_doc("kb-004", "Sepsis bundle first-hour actions", "a."))
        result = GenerateAnswerNode()(_make_state(ranked))
        assert isinstance(result["citations"], str)

    def test_gen_04_answer_is_grounded_in_ranked_passages_only(self):
        ranked = _ranked(
            _doc("kb-004", "Sepsis bundle first-hour actions", "obtain a lactate level within the first hour.")
        )
        answer = GenerateAnswerNode()(_make_state(ranked))["grounded_answer"]
        # Every content line traces to the single ranked passage.
        assert "obtain a lactate level within the first hour." in answer
        assert "[2]" not in answer


class TestNoCoverage:
    """Life-safety mandatory decline: no ranked passage -> explicit escalation
    text, never a fabricated / parametric answer (docs/02_design.md)."""

    def test_gen_05_empty_ranked_set_yields_no_coverage_answer(self):
        result = GenerateAnswerNode()(_make_state(_ranked()))
        assert "does not contain sufficient coverage" in result["grounded_answer"]
        assert "escalate to the clinical pharmacist" in result["grounded_answer"]
        assert from_json(result["citations"]) == []

    def test_missing_ranked_field_is_treated_as_no_coverage(self):
        state = _make_state(None)
        del state["ranked_documents"]
        result = GenerateAnswerNode()(state)
        assert "does not contain sufficient coverage" in result["grounded_answer"]

# HCR-C2-005 — Unit Tests: InputValidateNode (inner domain node 1)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller
# (inner Cat-2 domain node). Payloads are lowercase clinical phrasing free of
# patient identifiers and digit groups, so the framework input mask leaves them
# untouched — this is a life-safety domain.
#
# The caller-data contract itself (input_context channel, the finite/bounded
# numeric parser, the inert category grammar) is covered end-to-end in
# test_caller_data_contract.py.
#
# Mirrors docs/03_test_spec.md §2.2 (VAL-01..VAL-09).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.input_validate_node import InputValidateNode
from src.schemas.state import from_json


def _make_state(payload, **extra) -> dict:
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPlainTextParsing:
    def test_val_01_plain_text_becomes_query(self):
        result = InputValidateNode()(_make_state("sepsis bundle first hour actions"))
        assert result["search_query"] == "sepsis bundle first hour actions"
        filters = from_json(result["query_filters"])
        assert filters == {"category": None, "top_k": None, "score_threshold": None}

    def test_val_02_whitespace_is_collapsed(self):
        result = InputValidateNode()(_make_state("  sepsis   bundle\n first hour "))
        assert result["search_query"] == "sepsis bundle first hour"

    def test_query_filters_is_json_string(self):
        # Structured State fields travel as JSON strings, never dicts.
        result = InputValidateNode()(_make_state("sepsis bundle"))
        assert isinstance(result["query_filters"], str)
        assert isinstance(from_json(result["query_filters"]), dict)


class TestJsonEnvelopeParsing:
    def test_val_03_envelope_query_category_top_k(self):
        payload = json.dumps({"query": "venous thromboembolism prophylaxis", "category": "hematology", "top_k": 2})
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == "venous thromboembolism prophylaxis"
        filters = from_json(result["query_filters"])
        assert filters == {"category": "hematology", "top_k": 2, "score_threshold": None}

    def test_question_alias_accepted(self):
        payload = json.dumps({"question": "what is the vancomycin renal dose adjustment?"})
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == "what is the vancomycin renal dose adjustment?"

    def test_category_is_normalised(self):
        payload = json.dumps({"query": "penicillin allergy checks", "category": "  Allergy_Immunology "})
        result = InputValidateNode()(_make_state(payload))
        assert from_json(result["query_filters"])["category"] == "allergy_immunology"

    def test_val_04_malformed_json_falls_back_to_plain_text(self):
        payload = "{ this is not valid json but starts like it"
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == payload
        notes = from_json(result.get("intake_notes"), [])
        assert any("did not parse" in n for n in notes)


class TestTopKGuard:
    """VAL-05..07: the caller-supplied top_k is untrusted and fails CLOSED.

    A silently clamped or silently dropped parameter answers a question the
    caller did not ask; the request is refused instead, naming the field.
    """

    def test_val_05_out_of_range_top_k_is_rejected(self):
        for bad in (99, -5, 0):
            payload = json.dumps({"query": "sepsis bundle", "top_k": bad})
            result = InputValidateNode()(_make_state(payload))
            assert result["status"] == AgentStatus.SUCCESS.value
            assert "search_query" not in result
            assert any("top_k must be between" in entry for entry in result["error_log"])

    def test_val_06_non_numeric_top_k_is_rejected(self):
        payload = json.dumps({"query": "sepsis bundle", "top_k": "many"})
        result = InputValidateNode()(_make_state(payload))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert any("top_k must be a number" in entry for entry in result["error_log"])

    def test_val_07_fractional_top_k_is_rejected(self):
        payload = json.dumps({"query": "sepsis bundle", "top_k": 2.5})
        result = InputValidateNode()(_make_state(payload))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert any("whole number" in entry for entry in result["error_log"])


class TestSizeAndEmptyGuards:
    def test_val_08_oversize_query_is_truncated(self):
        payload = "guideline " * 300  # ~3300 chars after collapse
        result = InputValidateNode()(_make_state(payload))
        assert len(result["search_query"]) == 2000
        notes = from_json(result.get("intake_notes"), [])
        assert any("truncated" in n for n in notes)

    def test_val_09_empty_request_yields_note_not_error(self):
        result = InputValidateNode()(_make_state(""))
        assert result["search_query"] == ""
        notes = from_json(result.get("intake_notes"), [])
        assert any("empty request" in n for n in notes)

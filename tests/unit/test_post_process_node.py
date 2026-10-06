# HCR-C2-005 — Unit Tests: PostProcessNode (outer post_process slot; output gate)
#
# Invocation canon: node(state) via BaseNode.__call__. PostProcessNode is the
# second outer gate slot and requires VERIFIED_EXTERNAL (like PreProcessNode),
# so its behavioural tests build the state at that level; the ANONYMOUS
# rejection lives in test_trust_gate.py.
#
# Gate layering: the node's own module-level _security_gate_output() scan runs
# INSIDE execute() and replaces a violating answer with the sanitised stub
# (returned dict — no exception). The framework's own credential scan then sees
# only the clean stub. Intentional-credential tests assert the raw secret never
# survives into formatted_output OR result.
#
# The gate enforces three invariants, and each is tested BOTH ways — a
# violating output is blocked, and a compliant output passes byte-identical:
#   1. no disallowed content (credential-shaped or identifier-shaped strings)
#   2. the mandatory advisory disclaimer is present
#   3. every citation marker in the body resolves to a listed source
#
# Mirrors docs/03_test_spec.md §2.7 (POST-01..POST-09).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.output_format_node import _ADVISORY_DISCLAIMER
from src.nodes.post_process_node import PostProcessNode

_BODY = (
    "# Clinical Guideline Search Result\n\n"
    "[1] obtain a lactate level and blood cultures within the first hour of suspected septic shock.\n"
)

_SOURCES = "\n## Sources\n- [1] sepsis bundle (institutional protocol)\n"


def _report(body: str = _BODY, sources: str = _SOURCES, disclaimer: bool = True) -> str:
    """Compose a report in the shape OutputFormatNode emits."""
    tail = f"\n---\n\n*{_ADVISORY_DISCLAIMER}*" if disclaimer else ""
    return f"{body}{sources}{tail}"


_CLEAN_REPORT = _report()

# JWT-shaped token built at runtime so no credential-shaped literal ever sits
# in the repository (credential-scan hygiene).
_FAKE_JWT = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12


def _make_state(result_text, **extra) -> dict:
    state = {
        "result": result_text,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


def _assert_blocked(result, absent=None):
    assert result["status"] == AgentStatus.ERROR.value
    assert any("output blocked" in str(e) for e in result["error_log"])
    assert "[OUTPUT BLOCKED by the output gate" in result["formatted_output"]
    if absent is not None:
        # The offending value must not survive into either surfaced field.
        assert absent not in str(result.get("formatted_output", ""))
        assert absent not in str(result.get("result", ""))


class TestPostProcessClean:
    def test_post_01_clean_output_passes_through(self):
        result = PostProcessNode()(_make_state(_CLEAN_REPORT))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum.
        assert result["status"].__class__ is str
        assert result["formatted_output"] == _CLEAN_REPORT

    def test_post_02_empty_result_is_non_fatal(self):
        result = PostProcessNode()(_make_state(""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == ""


class TestDisallowedContentScan:
    def test_post_03_api_key_is_blocked(self):
        secret = "sk-ABCDEF0123456789abcdef"
        result = PostProcessNode()(_make_state(_report(body=f"{_BODY}debug api_key={secret}\n")))
        _assert_blocked(result, secret)

    def test_post_04_credential_assignment_is_blocked(self):
        secret = "super_secret_value_123"
        result = PostProcessNode()(_make_state(_report(body=f"{_BODY}internal note: password={secret}\n")))
        _assert_blocked(result, secret)

    def test_post_05_jwt_is_blocked(self):
        result = PostProcessNode()(_make_state(_report(body=f"{_BODY}session token {_FAKE_JWT}\n")))
        _assert_blocked(result, _FAKE_JWT)

    def test_post_06_bearer_token_is_blocked(self):
        secret = "Bearer abcdefghijklmnopqrstuvwxyz0123456789"
        result = PostProcessNode()(_make_state(_report(body=f"{_BODY}authorization: {secret}\n")))
        _assert_blocked(result, secret)

    @pytest.mark.parametrize(
        "identifier",
        [
            "MRN 4451233",
            "MRN-4451233",
            "123-45-6789",
        ],
    )
    def test_post_07_identifier_shaped_strings_are_blocked(self, identifier):
        # An identifier echoed back from caller input must never leave the
        # agent — and it must be REDACTED, not rewritten into a shape the scan
        # would no longer recognise. Nothing transforms the result before the
        # scan runs, so the whole output is replaced by the stub.
        result = PostProcessNode()(_make_state(_report(body=f"{_BODY}patient record {identifier}\n")))
        _assert_blocked(result, identifier)
        # The digits are gone entirely, not partially rewritten.
        assert "4451233" not in result["formatted_output"]
        assert "45-6789" not in result["formatted_output"]

    def test_nested_structures_are_scanned_at_every_level(self):
        nested = {"answer": _CLEAN_REPORT, "meta": [{"note": "sk-ABCDEF0123456789abcdef"}]}
        result = PostProcessNode()(_make_state(nested))
        _assert_blocked(result, "sk-ABCDEF0123456789abcdef")


class TestMandatoryDisclaimerEnforcement:
    def test_post_08_answer_without_the_disclaimer_is_blocked(self):
        # Fail-closed: if the domain pipeline ever stops attaching the advisory
        # disclaimer, the gate blocks the answer instead of shipping it bare.
        result = PostProcessNode()(_make_state(_report(disclaimer=False)))
        _assert_blocked(result)
        assert any("advisory disclaimer" in str(e) for e in result["error_log"])

    def test_disclaimer_present_passes(self):
        result = PostProcessNode()(_make_state(_report()))
        assert result["status"] == AgentStatus.SUCCESS.value


class TestCitationGroundingEnforcement:
    def test_post_09_unlisted_citation_marker_is_blocked(self):
        body = f"{_BODY}[2] a passage that is not listed in Sources.\n"
        result = PostProcessNode()(_make_state(_report(body=body)))
        _assert_blocked(result)
        assert any("Sources" in str(e) for e in result["error_log"])

    def test_answer_with_no_markers_is_grounded_by_definition(self):
        decline = (
            "# Clinical Guideline Search Result\n\n"
            "the clinical-guideline knowledge base does not contain sufficient coverage.\n"
        )
        result = PostProcessNode()(_make_state(_report(body=decline, sources="\n## Sources\n- none\n")))
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_every_marker_listed_passes(self):
        body = f"{_BODY}[2] a second cited passage.\n"
        sources = "\n## Sources\n- [1] sepsis bundle (protocol)\n- [2] second passage (protocol)\n"
        result = PostProcessNode()(_make_state(_report(body=body, sources=sources)))
        assert result["status"] == AgentStatus.SUCCESS.value

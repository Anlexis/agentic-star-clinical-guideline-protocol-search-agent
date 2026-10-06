# HCR-C2-005 — Unit Tests: the caller trust gate
#
# Contract: tests must invoke nodes via node(state) — through
# BaseNode.__call__, which runs the trust gate -> the input mask/injection
# screen -> execute() -> the output credential gate — never via
# node.execute(state) directly, which bypasses the gate. A denial RETURNS an
# error dict (never raises) with status ERROR and "trust gate denied" in
# error_log; execute() never runs, so execute-only output keys are ABSENT from
# the returned dict.
#
# The outer `main` slot is ClinicalGuidelineSearchGraphNode (a GraphNode
# wrapping the inner DomainWorkflowGraph), so the ANONYMOUS inner-node case
# below is exercised against InputValidateNode (the first inner domain node).
# TestTrustLevelMatrix asserts required_trust_level across all five inner
# domain nodes — the full trust matrix docs/02_design.md declares.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode, _ADVISORY_DISCLAIMER
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode


def _make_state(
    trust_value: str,
    user_input: str = "what is the current first-line empiric antibiotic for CAP under our stewardship protocol?",
    **extra,
) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": trust_value,
        "node_history": [],
        "error_log": [],
        "session_id": "test-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestTrustGate:
    """Trust gate tests — all invocations go through node(state) / __call__."""

    def test_anonymous_caller_allowed_on_inner_domain_node(self):
        """An ANONYMOUS caller passes an ANONYMOUS inner domain node
        (InputValidateNode — first node of the Cat 2 DomainWorkflowGraph)."""
        node = InputValidateNode()  # required_trust_level = ANONYMOUS
        result = node(_make_state(TrustLevel.ANONYMOUS.value, validated_input="test input"))
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert result.get("search_query") is not None

    def test_anonymous_caller_denied_on_pre_process(self):
        """Rejection: ANONYMOUS caller on the VERIFIED_EXTERNAL PreProcessNode.

        __call__ must RETURN an error dict (never raise) with status ERROR and
        'trust gate denied' in the error_log. execute() never ran, so the
        execute-only output key (validated_input) must be ABSENT.
        """
        node = PreProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(_make_state(TrustLevel.ANONYMOUS.value))
        assert result.get("status") == AgentStatus.ERROR.value
        error_log = result.get("error_log", [])
        assert any(
            "trust gate denied" in str(e) for e in error_log
        ), f"Expected 'trust gate denied' in error_log, got: {error_log}"
        assert "validated_input" not in result, "execute() must not run on a trust denial — validated_input leaked"

    def test_verified_external_caller_passes_pre_process(self):
        """A VERIFIED_EXTERNAL caller clears the pre_process gate and the node
        writes validated_input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value))
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("validated_input")

    def test_pre_process_empty_input_rejected_after_gate(self):
        """The gate passes, then the node's own validation rejects empty input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value, user_input=""))
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert any("empty" in str(e) for e in result.get("error_log", []))

    def test_anonymous_caller_denied_on_post_process(self):
        """Rejection on the other VERIFIED_EXTERNAL outer slot (post_process).

        The denial dict carries no execute-only key (formatted_output ABSENT).
        """
        node = PostProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(
            _make_state(
                TrustLevel.ANONYMOUS.value,
                result="a clean guideline answer about antibiotic stewardship",
            )
        )
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("trust gate denied" in str(e) for e in result.get("error_log", []))
        assert "formatted_output" not in result, "execute() must not run on a trust denial — formatted_output leaked"

    def test_verified_external_caller_passes_post_process(self):
        """A VERIFIED_EXTERNAL caller clears the post_process gate."""
        node = PostProcessNode()
        result = node(
            _make_state(
                TrustLevel.VERIFIED_EXTERNAL.value,
                result=("a clean guideline answer about antibiotic stewardship\n\n---\n\n" f"*{_ADVISORY_DISCLAIMER}*"),
            )
        )
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("formatted_output")


class TestTrustLevelMatrix:
    """The template's declared trust matrix (docs/02_design.md security section).

    Outer gate slots (pre_process / post_process) require VERIFIED_EXTERNAL
    (life-safety healthcare domain); the inner domain nodes of the Cat 2 nested
    DomainWorkflowGraph (input_validate / retrieve / rerank_filter /
    generate_answer / output_format) are ANONYMOUS behind that outer
    boundary — the external gate lives on the backbone, not the inner nodes.
    """

    def test_outer_gate_nodes_require_verified_external(self):
        assert PreProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
        assert PostProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL

    def test_inner_domain_nodes_admit_anonymous(self):
        for node_cls in (
            InputValidateNode,
            RetrieveNode,
            RerankFilterNode,
            GenerateAnswerNode,
            OutputFormatNode,
        ):
            assert node_cls.required_trust_level is TrustLevel.ANONYMOUS, (
                f"{node_cls.__name__} must declare TrustLevel.ANONYMOUS " "(inner Cat-2 domain node)"
            )

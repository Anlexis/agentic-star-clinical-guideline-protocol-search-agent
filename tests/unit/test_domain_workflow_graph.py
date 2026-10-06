# HCR-C2-005 — Unit Tests: DomainWorkflowGraph (inner BaseGraph)
#
# Inner-graph composition + a full inner invoke() over the seeded KB. The
# inner graph runs the 5 domain nodes (all ANONYMOUS) — the outer trust boundary
# is the AgentBaseGraph backbone's concern and is covered in
# test_graph_composition.py / the PoB suite.
#
# Config forwarding shape: _extra_initial_state() republishes the runtime
# retrieval block as THREE plain SCALAR state fields (retrieval_top_k /
# retrieval_score_threshold / retrieval_kb_path) — never a single JSON-string
# blob — because domain nodes read tuning straight off State, never through
# execute()'s signature. The same hook seeds the caller's input_context, which
# the framework does not forward across the outer→inner boundary.
#
# Mirrors docs/03_test_spec.md §3 (INT-01..INT-04).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from langgraph.graph import END

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus

from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import ClinicalGuidelineSearchGraphNode
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, from_json

# Same text as deploy/invoke_payload.json's "input" — lowercase, PII-free.
_CAP_QUERY = (
    "what is the current first-line empiric antibiotic for community-acquired "
    "pneumonia under our institutional stewardship protocol?"
)

# Zero token overlap with any seeded KB entry — drives the mandatory
# insufficient-evidence decline. Lowercase, free of patient identifiers.
_NO_COVERAGE_QUERY = "what is the recommended elevator inspection interval for hospital facilities management"


class TestInnerGraphConstruction:
    def test_int_01_inherits_base_graph(self):
        assert issubclass(DomainWorkflowGraph, BaseGraph)

    def test_int_01_registers_the_five_domain_nodes(self):
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert set(inner._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }
        assert isinstance(inner._nodes["input_validate"], InputValidateNode)
        assert isinstance(inner._nodes["retrieve"], RetrieveNode)
        assert isinstance(inner._nodes["rerank_filter"], RerankFilterNode)
        assert isinstance(inner._nodes["generate_answer"], GenerateAnswerNode)
        assert isinstance(inner._nodes["output_format"], OutputFormatNode)

    def test_inner_graph_name_and_schema(self):
        inner = DomainWorkflowGraph()
        assert inner.name == "hcr_c2_005_clinical_guideline_search_workflow"
        assert inner.state_schema is State

    def test_initialize_finalize_are_not_registered(self):
        # Outer backbone concerns must not leak into the inner topology.
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert "initialize" not in inner._nodes
        assert "finalize" not in inner._nodes


class TestConfigForwarding:
    def test_int_02_extra_initial_state_republishes_retrieval_as_scalars(self):
        inner = DomainWorkflowGraph(
            config={"configurable": {"retrieval": {"top_k": 2, "score_threshold": 0.6, "kb_path": "config/kb/x.json"}}}
        )
        extra = inner._extra_initial_state()
        assert extra == {
            "input_context": {},
            "retrieval_top_k": 2,
            "retrieval_score_threshold": 0.6,
            "retrieval_kb_path": "config/kb/x.json",
        }

    def test_extra_initial_state_with_no_config_seeds_only_the_context_channel(self):
        assert DomainWorkflowGraph()._extra_initial_state() == {"input_context": {}}

    def test_extra_initial_state_partial_retrieval_block_only_republishes_present_keys(self):
        inner = DomainWorkflowGraph(config={"configurable": {"retrieval": {"top_k": 3}}})
        assert inner._extra_initial_state() == {"input_context": {}, "retrieval_top_k": 3}

    def test_validate_config_rejects_a_broken_runtime_value(self):
        # A declared-but-broken runtime value must fail at compile time, not
        # degrade silently mid-run.
        for broken in (0, -1, "30", True, 1.5):
            inner = DomainWorkflowGraph(config={"configurable": {"timeout_seconds": broken}})
            try:
                inner._validate_config()
            except ValueError as exc:
                assert "timeout_s" in str(exc)
            else:  # pragma: no cover - the assertion below reports the miss
                raise AssertionError(f"broken timeout_s accepted: {broken!r}")
        DomainWorkflowGraph(config={"configurable": {"max_retry": 3, "timeout_seconds": 30}})._validate_config()


class TestOutputShape:
    def test_int_03_get_output_shapes_the_merge_contract(self):
        inner = DomainWorkflowGraph()
        out = inner.get_output(
            {
                "formatted_answer": "ANSWER",
                "citations": "[]",
                "status": AgentStatus.SUCCESS.value,
                "node_history": ["InputValidateNode"],
            }
        )
        assert out["formatted_answer"] == "ANSWER"
        assert out["citations"] == "[]"
        assert out["status"] == AgentStatus.SUCCESS.value
        assert out["node_history"] == ["InputValidateNode"]

    def test_route_returns_end_on_error(self):
        inner = DomainWorkflowGraph()
        assert inner.route({"status": AgentStatus.ERROR.value}) == END
        assert inner.route({"status": AgentStatus.SUCCESS.value}) == "output_format"


class TestInnerEndToEnd:
    def _invoke(self, payload: str) -> dict:
        # Same construction path the outer GraphNode uses: runtime config via
        # _parent_config(); domain nodes take NO ctor args.
        inner = DomainWorkflowGraph(config=ClinicalGuidelineSearchGraphNode()._parent_config())
        return inner.invoke(payload, session_id="inner-e2e")

    def test_int_04_full_inner_run_produces_the_formatted_answer(self):
        result = self._invoke(_CAP_QUERY)
        assert result["status"] == AgentStatus.SUCCESS.value
        answer = result["formatted_answer"]
        assert answer.startswith("# Clinical Guideline Search Result")
        assert "[1]" in answer
        assert "does not constitute a diagnosis, prescription, dosing order" in answer
        citations = from_json(result["citations"])
        assert citations and citations[0]["id"] == "kb-001"

    def test_int_04_inner_node_history_is_the_linear_topology(self):
        history = self._invoke(_CAP_QUERY)["node_history"]
        assert history == [
            "InputValidateNode",
            "RetrieveNode",
            "RerankFilterNode",
            "GenerateAnswerNode",
            "OutputFormatNode",
        ]

    def test_no_coverage_query_still_terminates_success(self):
        result = self._invoke(_NO_COVERAGE_QUERY)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result["formatted_answer"]

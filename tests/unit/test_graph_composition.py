# HCR-C2-005 — Unit Tests: nested Cat-2 graph composition (outer + end-to-end)
#
# Drives the REAL outer agent (ClinicalGuidelinesQAAgent / Graph) end-to-end
# via AgentBaseGraph.invoke(). The e2e context is
# InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL) — the
# manifest's declared caller level; for_internal() is NEVER used (it would
# over-privilege the run and hide trust-gate regressions).
#
# Covers BOTH life-safety-mandatory paths: the grounded-answer path
# (test_int_11_*) and the insufficient-evidence decline (no_coverage tests) —
# both must terminate SUCCESS end-to-end (declining is a successful, correct
# outcome, not an error).
#
# Mirrors docs/03_test_spec.md §3 (INT-05..INT-12).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pathlib

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

import src.graph.graph
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import (
    ClinicalGuidelineSearchGraphNode,
    ClinicalGuidelinesQAAgent,
    Graph,
    load_runtime_config,
)
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json, to_json

# Same text as deploy/invoke_payload.json's "input" — lowercase, PII-free.
_CAP_QUERY = (
    "what is the current first-line empiric antibiotic for community-acquired "
    "pneumonia under our institutional stewardship protocol?"
)

# Zero token overlap with any seeded KB entry — drives the mandatory
# insufficient-evidence decline. Lowercase, free of patient identifiers.
_NO_COVERAGE_QUERY = "what is the recommended elevator inspection interval for hospital facilities management"


def _run(
    user_input: str,
    trust: TrustLevel = TrustLevel.VERIFIED_EXTERNAL,
    input_context: dict | None = None,
) -> dict:
    ctx = InvocationContext(caller_trust_level=trust, caller_id="unit-suite")
    agent = Graph(config=load_runtime_config())
    return agent.invoke(user_input, ctx=ctx, input_context=input_context or {})


class TestOuterGraphConstruction:
    def test_int_05_inherits_agent_base_graph_directly(self):
        assert issubclass(ClinicalGuidelinesQAAgent, AgentBaseGraph)

    def test_int_05_graph_alias(self):
        assert Graph is ClinicalGuidelinesQAAgent

    def test_state_schema_is_state(self):
        assert ClinicalGuidelinesQAAgent().state_schema is State

    def test_int_06_compile_fills_all_backbone_slots(self):
        agent = ClinicalGuidelinesQAAgent()
        agent.compile()
        for slot in ("initialize", "pre_process", "main", "post_process", "finalize"):
            assert agent._nodes.get(slot) is not None, f"backbone slot not filled: {slot}"
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], ClinicalGuidelineSearchGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_add_edges_is_not_overridden(self):
        # Backbone wiring belongs to the framework — the template must not
        # redefine it.
        assert "add_edges" not in ClinicalGuidelinesQAAgent.__dict__


class TestMainSlotGraphNode:
    def test_int_07_get_subgraph_returns_the_inner_graph(self):
        subgraph = ClinicalGuidelineSearchGraphNode().get_subgraph()
        assert isinstance(subgraph, DomainWorkflowGraph)
        assert subgraph.config["configurable"]["retrieval"], "inner config must carry the retrieval block"

    def test_int_08_extract_input_prefers_validated_input(self):
        node = ClinicalGuidelineSearchGraphNode()
        assert node.extract_input({"validated_input": "VI", "user_input": "UI"}) == "VI"
        assert node.extract_input({"user_input": "UI"}) == "UI"

    def test_int_09_merge_output_maps_the_inner_contract(self):
        node = ClinicalGuidelineSearchGraphNode()
        citations = to_json([{"ref": 1, "id": "kb-001", "title": "t", "source": "s"}])
        delta = node.merge_output(
            {},
            {"formatted_answer": "ANSWER", "citations": citations, "status": AgentStatus.SUCCESS.value},
        )
        # The inner formatted_answer surfaces as BOTH guideline_answer and
        # result (PostProcessNode's output gate reads state["result"]).
        assert delta == {
            # nothing declined this run, so the reason slot crosses the
            # boundary empty rather than being dropped
            "error_code": "",
            "guideline_answer": "ANSWER",
            "result": "ANSWER",
            "citations": citations,
            "status": AgentStatus.SUCCESS.value,
        }

    def test_error_strategy_is_propagate_and_hitl_is_contained(self):
        assert ClinicalGuidelineSearchGraphNode.error_strategy == "propagate"
        assert ClinicalGuidelineSearchGraphNode.propagate_hitl is False

    def test_int_10_parent_config_never_empty_without_the_runtime_file(self, monkeypatch):
        # Even with an unreadable config/config.yaml the forwarded config
        # carries the fallback retrieval/llm blocks — never {}.
        monkeypatch.setattr(src.graph.graph, "_RUNTIME_CONFIG_PATH", pathlib.Path("/nonexistent/config.yaml"))
        cfg = ClinicalGuidelineSearchGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"]["kb_path"] == "config/kb/hcr_clinical_guideline_kb.json"
        assert cfg["configurable"]["retrieval"]["score_threshold"] == 0.75
        assert cfg["configurable"]["llm"]

    def test_runtime_values_reach_the_inner_graph_end_to_end(self, monkeypatch):
        # The declared runtime values must be LIVE, not decorative: a tightened
        # relevance floor in config/config.yaml has to change the answer the
        # agent produces, through both graph layers.
        monkeypatch.setattr(
            src.graph.graph,
            "load_runtime_config",
            lambda: {
                "max_retry": 3,
                "timeout_s": 30,
                "retrieval": {
                    "top_k": 8,
                    "score_threshold": 0.99,
                    "kb_path": "config/kb/hcr_clinical_guideline_kb.json",
                },
                "llm": {"temperature": 0.0, "max_tokens": 4000},
            },
        )
        result = _run(_CAP_QUERY)
        assert result.get("status") == AgentStatus.SUCCESS.value
        # Nothing clears a 0.99 floor, so the run declines instead of answering.
        assert "does not contain sufficient coverage" in result.get("output", "")

    def test_broken_runtime_value_fails_at_compile_time(self, monkeypatch):
        monkeypatch.setattr(
            src.graph.graph,
            "load_runtime_config",
            lambda: {"max_retry": 3, "timeout_s": 0},
        )
        agent = ClinicalGuidelinesQAAgent(config=src.graph.graph.load_runtime_config())
        try:
            agent.compile()
        except ValueError as exc:
            assert "timeout_s" in str(exc)
        else:  # pragma: no cover - the assertion below reports the miss
            raise AssertionError("a non-positive timeout_s was accepted")


class TestEndToEndInvoke:
    """Full agent run: outer backbone + inner domain workflow, no LLM."""

    def test_int_11_invoke_returns_success(self):
        result = _run(_CAP_QUERY)
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected success, got {result.get('status')}. result={result!r}"

    def test_int_11_output_is_the_gated_formatted_answer(self):
        output = _run(_CAP_QUERY).get("output")
        assert isinstance(output, str) and output.strip()
        assert output.startswith("# Clinical Guideline Search Result")
        assert "[1]" in output
        assert "does not constitute a diagnosis, prescription, dosing order" in output

    def test_int_11_e2e_traverses_the_post_process_gate(self):
        history = _run(_CAP_QUERY).get("node_history", [])
        for cls_name in ("PreProcessNode", "ClinicalGuidelineSearchGraphNode", "PostProcessNode"):
            assert cls_name in history, f"node_history missing {cls_name}: {history}"

    def test_no_coverage_query_still_terminates_success(self):
        """Life-safety mandatory decline: SUCCESS status, explicit escalation
        text — never an ERROR, never a fabricated answer."""
        result = _run(_NO_COVERAGE_QUERY)
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result.get("output", "")

    def test_int_12_anonymous_caller_is_denied_at_the_outer_boundary(self):
        """Trust gate at graph level: an ANONYMOUS invoke is refused by the
        VERIFIED_EXTERNAL pre_process slot. The error state short-circuits the
        main slot and routes past post_process to finalize — no domain answer
        is ever produced."""
        result = _run(_CAP_QUERY, trust=TrustLevel.ANONYMOUS)
        assert result.get("status") == AgentStatus.ERROR.value
        assert not result.get("output")
        history = result.get("node_history", [])
        assert "PostProcessNode" not in history
        assert history[:2] == ["InitializeNode", "PreProcessNode"]


class TestStateRoundTrip:
    """State helpers: producers to_json() on write, consumers from_json()."""

    def test_to_from_json_list_round_trip(self):
        original = [{"id": "kb-001", "score": 0.83, "title": "cap empiric antibiotic"}]
        assert from_json(to_json(original)) == original

    def test_to_from_json_dict_round_trip(self):
        original = {"category": "infectious_disease", "top_k": 3}
        assert from_json(to_json(original)) == original

    def test_to_json_none_passes_through(self):
        assert to_json(None) is None

    def test_from_json_malformed_returns_default(self):
        assert from_json("{not valid json", default=[]) == []
        assert from_json(None, default={}) == {}
        assert from_json("", default=[]) == []

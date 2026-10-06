# PB-6 - Invoke-Order Boundary: a full agent.invoke() must execute the fixed
# AgentBaseGraph backbone in order.
#
# The Cat 1 backbone is fixed and is NEVER overridden by a Cat 2 template
# (add_edges() belongs to the framework):
#
#     START -> initialize -> pre_process -> main -> {route} -> post_process
#           -> finalize -> END
#
# The framework records every executed node in `node_history` (an AgentState
# field whose reducer is operator.add, so entries accumulate in execution
# order). Each entry is the node's CLASS NAME - appended by BaseNode.__call__.
#
# For HCR-C2-005 (Cat 2, two-layer nested) the `main` slot is a GraphNode
# subclass (ClinicalGuidelineSearchGraphNode) that delegates to the inner
# DomainWorkflowGraph. The inner graph runs with its own state; its inner
# node_history is NOT merged back into the outer state (merge_output() maps
# only guideline_answer / result / citations / status), so the OUTER
# node_history contains exactly the five backbone slots - never the inner
# domain nodes.
#
# This test drives a real end-to-end Graph().invoke() over the standard deploy
# payload and asserts the surfaced node_history matches the canonical backbone
# order. A SUCCESS terminal status is required: on any non-SUCCESS status
# route() short-circuits main -> finalize and the post_process slot is skipped,
# which is itself an invoke-order violation this test would catch.
#
# Trust context: InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
# - the manifest's declared caller level. for_internal() is NEVER used here:
# it would over-privilege the run and hide trust-gate regressions on the outer slot.
#
# docs/03_test_spec.md section 4 (PoB).
# Deterministic - no LLM, no network. framework.* / src.* imports only.

import json
import pathlib

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import Graph

# --- TEMPLATE-SPECIFIC ------------------------------------------------------
# The `main`-slot GraphNode class name for THIS template. A sibling template
# mirroring this canonical changes ONLY this one entry (its own domain
# <...>GraphNode); the other four backbone slot names are framework-fixed
# fixed and identical across every Cat 1 / Cat 2 template.
_MAIN_SLOT_NODE = "ClinicalGuidelineSearchGraphNode"

# The valid, PII-free domain payload that drives the full clinical-guideline
# search workflow to a SUCCESS terminal status. MUST stay byte-equal to the
# `input` field of deploy/invoke_payload.json (the standard deploy payload) -
# enforced by test_payload_matches_deploy_invoke_payload below.
_VALID_PAYLOAD = (
    "What is the current first-line empiric antibiotic for community-acquired "
    "pneumonia under our institutional stewardship protocol?"
)

_DEPLOY_PAYLOAD_PATH = pathlib.Path(__file__).resolve().parents[2] / "deploy" / "invoke_payload.json"
# --- END TEMPLATE-SPECIFIC --------------------------------------------------

# Canonical AgentBaseGraph backbone execution order, by node class name as
# recorded in node_history. Four entries are framework-fixed and
# identical for every template; only _MAIN_SLOT_NODE is template-specific.
_EXPECTED_ORDER = [
    "InitializeNode",  # framework default  (initialize slot)
    "PreProcessNode",  # standard  (pre_process slot: trust + input screens)
    _MAIN_SLOT_NODE,  # TEMPLATE-SPECIFIC  (main slot GraphNode)
    "PostProcessNode",  # standard  (post_process slot: the output gate)
    "FinalizeNode",  # framework default  (finalize slot)
]


def _run() -> dict:
    """Run a full end-to-end invocation at the manifest's declared trust level."""
    ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, caller_id="pb6-suite")
    return Graph().invoke(_VALID_PAYLOAD, ctx=ctx)


class TestInvokeOrderBoundary:
    """PB-6: full agent.invoke() executes the backbone in the fixed order."""

    def test_payload_matches_deploy_invoke_payload(self):
        """_VALID_PAYLOAD must be byte-equal to deploy/invoke_payload.json's
        `input` - PB-6 proves invoke-order for the SAME payload the deployment
        smoke test sends."""
        deployed = json.loads(_DEPLOY_PAYLOAD_PATH.read_text(encoding="utf-8"))
        assert _VALID_PAYLOAD == deployed["input"]

    def test_invoke_reaches_success(self):
        """The full run must terminate SUCCESS - otherwise route() short-circuits
        main -> finalize and the post_process slot never runs."""
        result = _run()
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected SUCCESS, got {result.get('status')!r}. result={result!r}"

    def test_output_is_non_empty(self):
        """A successful run must surface a non-empty gated output."""
        assert _run().get("output"), "invoke() surfaced an empty output"

    def test_node_history_is_populated(self):
        """node_history must be a non-empty list of node class-name strings."""
        history = _run().get("node_history")
        assert isinstance(history, list) and history, f"node_history must be a non-empty list, got {history!r}"
        assert all(isinstance(n, str) for n in history), f"node_history entries must be strings, got {history!r}"

    def test_backbone_slot_order(self):
        """Core invoke-order boundary: the pre_process slot runs before the domain
        main slot, which runs before the post_process slot - as a strict ordered
        subsequence of node_history."""
        history = _run().get("node_history", [])
        ordered_slots = ["PreProcessNode", _MAIN_SLOT_NODE, "PostProcessNode"]
        for name in ordered_slots:
            assert name in history, f"Expected backbone slot {name!r} in node_history, got {history!r}"
        positions = [history.index(name) for name in ordered_slots]
        assert positions == sorted(positions), (
            f"Backbone slots executed out of order: {ordered_slots} at {positions}. " f"node_history={history!r}"
        )

    def test_full_backbone_sequence(self):
        """The complete AgentBaseGraph backbone order:
        initialize -> pre_process -> main -> post_process -> finalize."""
        history = _run().get("node_history", [])
        assert history == _EXPECTED_ORDER, (
            "node_history does not match the canonical backbone order.\n"
            f"  expected: {_EXPECTED_ORDER}\n"
            f"  actual:   {history}"
        )

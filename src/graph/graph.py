"""AgentCore Platform v1.0"""

# HCR-C2-005 - Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Clinical Guidelines & Protocol KB Search Agent (Cat 2 RAG domain workflow).
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed - identical to Cat 1, do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
#                                             |  (RETRY, max_retry from config/config.yaml)
#                                             -> pre_process
#
#   `main` slot is a GraphNode subclass (ClinicalGuidelineSearchGraphNode)
#   that delegates the full clinical-guideline search domain workflow to
#   DomainWorkflowGraph (inner BaseGraph: input_validate -> retrieve ->
#   rerank_filter -> generate_answer -> output_format).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#   src/graph/context_bridge.py        <- input_context outer->inner hand-off
#
# Class-name contract:
#   graph.py class:           ClinicalGuidelinesQAAgent (this file)
#   config/agent.yaml class:  "src.graph.graph.ClinicalGuidelinesQAAgent"  <- must match
#   src/api/server.py import: from src.graph.graph import ClinicalGuidelinesQAAgent
#
# Rules enforced:
#   - ClinicalGuidelinesQAAgent inherits AgentBaseGraph (framework base class -
#     direct inheritance)
#   - super().register_nodes() called first (fills initialize + finalize)
#   - ClinicalGuidelineSearchGraphNode assigned to self._nodes["main"]
#   - _parent_config() forwards the config/config.yaml runtime blocks (never {})
#   - merge_output() returns only changed keys
#   - add_edges() NOT overridden on the outer graph
#   - No platform-internal SDK imports

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from framework.schemas.agent_status import AgentStatus
from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from src.graph.context_bridge import set_caller_input_context
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State

if TYPE_CHECKING:
    from src.graph.domain_workflow_graph import DomainWorkflowGraph

# Runtime-parameter file: src/graph/graph.py -> parents[2] = repo root.
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

# Fallbacks mirror the `retrieval` / `llm` blocks in config/config.yaml so
# _parent_config() never forwards an empty config even if the file is
# unreadable in an exotic deployment layout. Healthcare life-safety floor:
# score_threshold 0.75, temperature 0.0.
_FALLBACK_RETRIEVAL: dict[str, Any] = {
    "top_k": 8,
    "score_threshold": 0.75,
    "kb_path": "config/kb/hcr_clinical_guideline_kb.json",
}
_FALLBACK_LLM: dict[str, Any] = {
    "temperature": 0.0,
    "max_tokens": 4000,
}


def load_runtime_config() -> dict[str, Any]:
    """Read the runtime parameters from config/config.yaml.

    Returns an empty dict — never raises — when the file is absent,
    unreadable, not valid YAML, or not a mapping. Keys are returned exactly as
    the file declares them (`timeout_s`, not `timeout_seconds`); the mapping to
    the key the inner graph validates happens in `_parent_config()`.

    src/api/server.py passes the result into the agent constructor so the
    declared `max_retry` is live in the backbone's retry routing instead of
    silently falling back to the framework default.
    """
    try:
        import yaml

        loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


class ClinicalGuidelineSearchGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of the outer agent.

    Wraps DomainWorkflowGraph (inner Cat 2 BaseGraph RAG pipeline).
    Called by AgentBaseGraph backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()    - instantiate DomainWorkflowGraph with the forwarded
                          runtime config (_parent_config())
      extract_input()   - pull validated_input (identifier-stripped) from outer
                          state; stash input_context for the inner graph
      merge_output()    - map sub_result fields into outer state delta (changed keys only)
      error_strategy    - "propagate": re-raise inner errors as SubgraphError (fail-fast)
    """

    # "propagate": re-raise inner graph exceptions as SubgraphError (default - fail fast).
    # "handle": call on_subgraph_error() instead - use for graceful degradation.
    error_strategy: ClassVar[str] = "propagate"

    # False: HITL interrupts are handled inside the inner graph only (this
    # template has no HITL path).
    propagate_hitl: ClassVar[bool] = False

    def _parent_config(self) -> dict[str, Any]:
        """Forward the config/config.yaml runtime blocks to the inner graph.

        Loads config/config.yaml and returns the tuning blocks under
        config["configurable"] - never an empty dict. The inner graph
        republishes the `retrieval` block into inner state
        (DomainWorkflowGraph._extra_initial_state()) so RetrieveNode /
        RerankFilterNode read live top_k / score_threshold / kb_path values
        instead of dead declarations. The `llm` block is forwarded verbatim for
        the documented LLM-synthesis upgrade (unused by the deterministic
        pipeline). `max_retry` / `timeout_s` travel too - `timeout_s` renamed to
        `timeout_seconds`, the key the inner graph validates - so a
        declared-but-broken runtime value fails loudly at compile time.
        """
        loaded = load_runtime_config()
        retrieval = loaded.get("retrieval")
        if not isinstance(retrieval, dict) or not retrieval:
            retrieval = dict(_FALLBACK_RETRIEVAL)
        llm = loaded.get("llm")
        if not isinstance(llm, dict) or not llm:
            llm = dict(_FALLBACK_LLM)
        configurable: dict[str, Any] = {"retrieval": retrieval, "llm": llm}
        if "max_retry" in loaded:
            configurable["max_retry"] = loaded["max_retry"]
        if "timeout_s" in loaded:
            configurable["timeout_seconds"] = loaded["timeout_s"]
        return {"configurable": configurable}

    def get_subgraph(self) -> "DomainWorkflowGraph":
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time.

        The inner graph receives the runtime config via its BaseGraph ctor; its
        domain NODES still take no constructor arguments and read retrieval
        tuning per-call from State (execute(self, state) only - no config
        parameter).
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def execute(self, state: AgentState) -> dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request declined by pre_process has no validated input to act on, so
        running the inner graph would only produce a second, vaguer reason for
        the same rejection - and overwrite the specific one already settled.
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        result: dict[str, Any] = super().execute(state)
        return result

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        PreProcessNode validates and identifier-strips the raw user_input and
        writes the result to validated_input. Prefer that; fall back to
        user_input if validated_input is absent (e.g. in unit tests).
        Structured params (category / top_k) may also travel inside this string
        as a JSON envelope and are parsed back by the first inner node
        (InputValidateNode).

        Side effect (deliberate): the caller's input_context is stashed on the
        context bridge here, immediately before the framework invokes the inner
        graph without it. DomainWorkflowGraph._extra_initial_state() reads it
        back while building the inner initial state.
        """
        set_caller_input_context(state.get("input_context") or {})
        payload = state.get("validated_input") or state.get("user_input", "")
        return str(payload) if payload else ""

    def merge_output(self, state: AgentState, sub_result: dict[str, Any]) -> dict[str, Any]:
        """Map inner graph sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys - never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  -> "formatted_answer", "citations", "status", ...
          This merge_output() reads -> sub_result.get("formatted_answer"),
                                       sub_result.get("citations"),
                                       sub_result.get("status")

        guideline_answer (str | None): final rendered clinical-guideline
          answer; written by OutputFormatNode inside the inner graph.
        result: PostProcessNode (outer post_process slot) reads
          state.get("result") - the inner graph emits the rendered answer
          under "formatted_answer", so map it to "result" as well; otherwise
          the final output surfaced by PostProcessNode (and its output gate) is
          always empty.
        status (str | None): terminal AgentStatus value from the inner graph run.
        """
        return {
            # Outer reason wins: a reason settled before the inner run is the real
            # one, and a plain sub_result.get() would erase it.
            "error_code": state.get("error_code") or sub_result.get("error_code", ""),
            "guideline_answer": sub_result.get("formatted_answer"),
            "result": sub_result.get("formatted_answer"),
            "citations": sub_result.get("citations"),
            "status": sub_result.get("status"),
        }


class ClinicalGuidelinesQAAgent(AgentBaseGraph):
    """Outer graph for HCR-C2-005 (Cat 2 RAG).

    Inherits AgentBaseGraph directly (framework base class). Domain logic is
    fully encapsulated in ClinicalGuidelineSearchGraphNode (main slot), which
    delegates to DomainWorkflowGraph (inner BaseGraph).

    Backbone (fixed - identical to Cat 1):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() is the ONLY override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process:  PreProcessNode (input validation + identifier strip)
      - main:         ClinicalGuidelineSearchGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output gate)

    add_edges() is NOT overridden - backbone wiring belongs to the framework.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with the platform registry."""
        return "ClinicalGuidelinesQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def _validate_config(self) -> None:
        """Validate the runtime config, then add this template's own rule.

        The framework base validates `max_retry` / `memory_enabled` / `hitl`.
        `timeout_s` is declared in config/config.yaml but not framework-known,
        so it is validated here: present-but-broken fails loudly at compile
        time rather than degrading silently.
        """
        super()._validate_config()
        timeout_s = self.config.get("timeout_s")
        if timeout_s is not None and (not isinstance(timeout_s, int) or isinstance(timeout_s, bool) or timeout_s <= 0):
            raise ValueError(
                f"[{self.__class__.__name__}] config/config.yaml 'timeout_s' must be a "
                f"positive integer, got {timeout_s!r}"
            )

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first - it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = ClinicalGuidelineSearchGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden - backbone wiring belongs to the framework.


# Back-compat alias - config/agent.yaml declares
# class: "src.graph.graph.ClinicalGuidelinesQAAgent", and src/api/server.py
# imports the class directly. Keep both names pointing at the agent.
Graph = ClinicalGuidelinesQAAgent

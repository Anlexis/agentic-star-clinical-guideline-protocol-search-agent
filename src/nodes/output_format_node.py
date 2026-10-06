"""AgentCore Platform v1.0"""

# HCR-C2-005 - OutputFormatNode
# Domain node 5 (terminal): compose the final formatted answer - the grounded
# answer body, the Sources list, and the mandatory advisory disclaimer. The
# disclaimer is part of THIS node's domain output contract (golden-transform
# parity with a peer template's OutputFormatNode / advisory-disclaimer pattern):
# it is a fixed constant, appended unconditionally to EVERY answer (including
# the insufficient-evidence decline) - never driven by a prompt, config
# value, or caller input, so it cannot be suppressed. The outer post_process
# gate does not compose the disclaimer; it independently VERIFIES that this
# node attached it and blocks the output if it is missing.
#
# execute(self, state) -> dict only - NO config parameter.
#
# Wired by the inner graph (DomainWorkflowGraph). get_output() of the inner
# graph surfaces formatted_answer + status to the outer merge_output().
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json

# Mandatory advisory disclaimer - appended to EVERY answer this template
# emits (including the insufficient-evidence decline). Hardcoded, not
# caller-configurable via prompt or config, and independently verified by the
# output gate in src/nodes/post_process_node.py.
_ADVISORY_DISCLAIMER = (
    "Advisory only - physician/clinician judgment required. This answer is "
    "generated from the seeded clinical-guideline knowledge base for "
    "advisory reference only. It does not constitute a diagnosis, "
    "prescription, dosing order, or a substitute for clinical judgment. "
    "Verify against the primary guideline or protocol document and your "
    "institution's clinical decision-support policy before acting on it."
)


class OutputFormatNode(FunctionNode):
    """Compose the final answer: body + sources + advisory disclaimer.

    Input state keys:
        grounded_answer: answer body with [n] citation markers, or the
                          insufficient-evidence decline
        citations:        JSON list [{ref, id, title, source}]

    Output state keys (partial dict):
        formatted_answer: final rendered answer string
        status:           AgentStatus.SUCCESS.value (plain string — never write
                          the bare enum to State)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        grounded_answer = state.get("grounded_answer") or ("No answer is available for this request.")
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []

        lines: List[str] = []
        lines.append("# Clinical Guideline Search Result")
        lines.append("")
        lines.append(grounded_answer)
        lines.append("")
        lines.append("## Sources")
        if citations:
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                ref = citation.get("ref", "?")
                title = str(citation.get("title", "")).strip()
                source = str(citation.get("source", "")).strip()
                suffix = f" ({source})" if source else ""
                lines.append(f"- [{ref}] {title}{suffix}")
        else:
            lines.append("- none (no knowledge-base passage cleared the relevance threshold)")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*{_ADVISORY_DISCLAIMER}*")

        formatted_answer = "\n".join(lines)

        # Domain audit: final answer composed (disclaimer attached).
        emit_trace_event(
            "output_format_complete",
            {
                "answer_chars": len(formatted_answer),
                "citation_count": len(citations),
            },
            state,
        )

        return {
            "formatted_answer": formatted_answer,
            "status": AgentStatus.SUCCESS.value,
        }

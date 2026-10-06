"""AgentCore Platform v1.0"""

# HCR-C2-005 - GenerateAnswerNode
# Domain node 4: assemble the grounded answer from the ranked clinical
# guideline passages, or return the explicit insufficient-evidence decline
# when no passage cleared the RerankFilterNode relevance floor.
#
# v1 is DETERMINISTIC (no live LLM call): the answer is rule-assembled from
# the ranked passages only - a lead sentence plus one cited point per
# passage, each carrying a numbered citation marker [n]. Apart from the
# caller's own question, which InputValidateNode has already reduced to a
# single line of inert text, nothing outside the ranked_documents input
# reaches the answer body, so the output is grounded by construction - this is
# how the template satisfies "answers must be strictly grounded". The LLM
# synthesis upgrade seam is documented in docs/02_design.md ("v1
# Implementation Note - LLM synthesis") and
# config/prompts/answer_synthesis_prompt.md: a v2 node swaps the assembly for
# an LLM call over the same input and emits the same state contract.
#
# execute(self, state) -> dict only - NO config parameter.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.schemas.agent_status import AgentStatus
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json

# Explicit insufficient-evidence decline - shown when no KB passage cleared
# the life-safety relevance floor (config/config.yaml score_threshold).
# Never falls back to parametric/unsourced knowledge.
_NO_COVERAGE_ANSWER = (
    "The clinical-guideline knowledge base does not contain sufficient "
    "coverage to answer this question with confidence. Rephrase the query "
    "with more specific protocol, guideline, or drug terms, or escalate to "
    "the clinical pharmacist / quality-and-safety team for manual review. "
    "Do not use this system to make treatment, dosing, or diagnostic "
    "decisions when guidance is unavailable."
)

# Cited excerpt length per passage inside the answer body.
_POINT_EXCERPT_CHARS = 240


def _first_sentences(text: str, limit: int) -> str:
    """Trim an excerpt at a sentence boundary where possible, else hard-cap."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    period = cut.rfind(". ")
    if period > limit // 2:
        return cut[: period + 1]
    return cut.rstrip() + "..."


class GenerateAnswerNode(FunctionNode):
    """Rule-based grounded answer assembly with numbered citations.

    Input state keys:
        ranked_documents: JSON list of surviving passages (from RerankFilterNode)
        search_query:     normalised question (for the lead sentence)

    Output state keys (partial dict):
        grounded_answer: answer body with [n] citation markers, or the
                          insufficient-evidence decline
        citations:        JSON list [{ref, id, title, source}]
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        ranked: List[Dict[str, Any]] = from_json(state.get("ranked_documents"), []) or []
        query = state.get("search_query") or ""

        citations: List[Dict[str, Any]] = []

        if not ranked:
            grounded_answer = _NO_COVERAGE_ANSWER
        else:
            lines: List[str] = []
            if query:
                lines.append(
                    f"Based on the seeded clinical-guideline knowledge base, "
                    f'the following passages answer the question: "{query}"'
                )
            else:
                lines.append(
                    "Based on the seeded clinical-guideline knowledge base, " "the most relevant passages are:"
                )
            lines.append("")
            for ref, doc in enumerate(ranked, start=1):
                if not isinstance(doc, dict):
                    continue
                title = str(doc.get("title", "")).strip()
                excerpt = _first_sentences(str(doc.get("excerpt", "")), _POINT_EXCERPT_CHARS)
                lines.append(f"[{ref}] {title}: {excerpt}")
                citations.append(
                    {
                        "ref": ref,
                        "id": str(doc.get("id", "")),
                        "title": title,
                        "source": str(doc.get("source", "")),
                    }
                )
            grounded_answer = "\n".join(lines)

        # Domain audit: grounded answer assembled (or declined).
        emit_trace_event(
            "generate_answer_complete",
            {
                "citation_count": len(citations),
                "answer_chars": len(grounded_answer),
                "no_coverage": not ranked,
            },
            state,
        )

        return {
            "grounded_answer": grounded_answer,
            "citations": to_json(citations),
        }

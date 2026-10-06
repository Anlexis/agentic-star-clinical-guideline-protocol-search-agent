"""AgentCore Platform v1.0"""

# HCR-C2-005 - PostProcessNode (outer post_process slot; the output gate)
#
# This node calls the MODULE-LEVEL `_security_gate_output()` scan from
# execute() itself. No `_extra_security_gate_input` / `_extra_security_gate_output`
# instance methods are defined here (the framework auto-wraps such hooks into
# the graph chain).
#
# The gate enforces the three invariants this template's answers state, and it
# enforces them for EVERY representation of the result - `content` may be a
# plain string OR a nested structure (dict/list/tuple), and the scan walks every
# level and checks every string leaf, so a violation buried inside a nested
# field is caught exactly like a top-level string:
#
#   1. DISALLOWED CONTENT - credential-shaped strings (API keys, JWTs, Bearer
#      tokens, credential assignments) and identifier-shaped strings (medical
#      record numbers, national-identifier-shaped digit groups) never leave the
#      agent. A hit replaces the whole output with a sanitised stub.
#   2. MANDATORY ADVISORY DISCLAIMER - every non-empty answer carries the
#      advisory line OutputFormatNode composes. The gate does not compose it;
#      it VERIFIES it, so if a future change to the domain pipeline ever drops
#      the disclaimer the output is blocked rather than shipped bare.
#   3. CITATION GROUNDING - every numbered citation marker in the answer body
#      resolves to an entry in the rendered Sources list. An answer that cites
#      a passage it does not list is blocked.
#
# LAYER ORDER (deliberate, do not reorder): the pattern scan runs FIRST, on the
# untouched result, and any later transformation of the output must re-run it.
# A transformation that rewrites characters inside a matched span can destroy
# the very shape the scan keys on - a redaction is order-independent only when
# nothing rewrites the text before it.
#
# This template renders no monetary or statistical aggregates: the answer body
# is composed exclusively of knowledge-base passage excerpts plus the caller's
# own question, so there is no rounding/precision invariant to enforce, and
# numeric content (drug doses, ICD codes, protocol numbers, guideline revision
# dates) is reproduced verbatim by design - rewriting a dose would be a far
# worse clinical defect than any precision gain.

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, TOO_LONG
from src.nodes.output_format_node import _ADVISORY_DISCLAIMER

logger = logging.getLogger(__name__)

# Disallowed output-content patterns. Each tuple: (name, compiled regex) —
# order matters (most specific first).
_DISALLOWED_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    # API key patterns: sk-..., pk-..., ak-...
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    # JWT: three base64url segments separated by dots
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    # Bearer token in an Authorization-like context
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/]{20,}", re.IGNORECASE)),
    # Credential assignment patterns
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
    # Identifier-shaped tokens. The knowledge base carries published guideline
    # text and no patient data, so these are a defensive floor against an
    # identifier echoed back from caller-supplied input: a medical-record-number
    # style token, or a 3-2-4 national-identifier-shaped digit group.
    ("mrn_like_id", re.compile(r"\bMRN[-:\s]?\d{6,10}\b", re.IGNORECASE)),
    ("national_id_like", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
]

_SANITISED_STUB = (
    "[OUTPUT BLOCKED by the output gate - disallowed content detected. "
    "Review the generated answer and retry without credential-like or "
    "identifier-like strings.]"
)

_DISCLAIMER_STUB = (
    "[OUTPUT BLOCKED by the output gate - the mandatory advisory disclaimer "
    "was not present on the generated answer.]"
)

_GROUNDING_STUB = (
    "[OUTPUT BLOCKED by the output gate - the generated answer cited a passage "
    "that is not listed in its Sources section.]"
)

# Heading that opens the rendered Sources list (OutputFormatNode).
_SOURCES_HEADING = "## Sources"

# Numbered citation marker, e.g. "[3]".
_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")

# A rendered Sources row, e.g. "- [3] Title (source)".
_SOURCE_ROW_RE = re.compile(r"^-\s*\[(\d+)\]", re.MULTILINE)


def _security_gate_output(content: Any) -> Optional[str]:
    """Run the disallowed-content scan, RECURSIVELY.

    `content` may be a str, or a dict/list/tuple that nests strings at any
    depth. Every string leaf is scanned — a violation nested inside a
    dict/list is caught exactly like a top-level string.

    Returns the name of the first matched violation, or None if clean.
    """
    if isinstance(content, str):
        for name, pattern in _DISALLOWED_PATTERNS:
            if pattern.search(content):
                return name
        return None
    if isinstance(content, dict):
        for value in content.values():
            violation = _security_gate_output(value)
            if violation:
                return violation
        return None
    if isinstance(content, (list, tuple)):
        for item in content:
            violation = _security_gate_output(item)
            if violation:
                return violation
        return None
    # Non-string scalars (int/float/bool/None/...) carry no disallowed text.
    return None


def _citations_are_grounded(text: str) -> bool:
    """True when every citation marker in the answer body is listed in Sources.

    The body is everything before the Sources heading (the whole text when no
    Sources section was rendered — an answer that cites without listing is a
    grounding failure either way).
    """
    head, sep, tail = text.partition(_SOURCES_HEADING)
    body_refs = set(_CITATION_MARKER_RE.findall(head))
    if not body_refs:
        return True
    listed_refs = set(_SOURCE_ROW_RE.findall(tail)) if sep else set()
    return body_refs.issubset(listed_refs)


# Reason code -> the sentence the caller reads. A code with no entry falls
# back to the generic one rather than leaking the code itself.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Format and finalize the output, behind the output gate."""

    # Explicit by design, not inherited implicitly. Outer backbone gate slot —
    # matches the manifest's declared required_trust_level (config/agent.yaml).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _blocked(self, reason: str, stub: str) -> Dict[str, Any]:
        """Fail-closed block: replace the output and report the reason."""
        logger.error("PostProcessNode: OUTPUT BLOCKED - %s", reason)
        return {
            "formatted_output": stub,
            "result": stub,
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PostProcessNode: output blocked - {reason}"],
        }

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # A run declined upstream has nothing to format. Render the reason as
        # the caller-facing body and carry the marker onward.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
                "result": message,
                "formatted_output": message,
            }
        result = state.get("result", "")

        if not result or (isinstance(result, str) and not result.strip()):
            # No result to gate — forward as-is (non-fatal).
            return {
                "formatted_output": result,
                "status": AgentStatus.SUCCESS.value,
            }

        # Layer 1 — disallowed-content scan, on the untouched result.
        violation = _security_gate_output(result)
        if violation:
            return self._blocked(f"disallowed content detected ({violation})", _SANITISED_STUB)

        # Layer 2 — the mandatory advisory disclaimer must be present.
        if isinstance(result, str) and _ADVISORY_DISCLAIMER not in result:
            return self._blocked("mandatory advisory disclaimer missing", _DISCLAIMER_STUB)

        # Layer 3 — every citation marker must resolve to a listed source.
        if isinstance(result, str) and not _citations_are_grounded(result):
            return self._blocked("citation marker not listed in the Sources section", _GROUNDING_STUB)

        # Domain audit: a finalized answer was emitted.
        emit_trace_event(
            "post_process_complete",
            {"output_chars": len(str(result))},
            state,
        )

        return {
            "formatted_output": result,
            "status": AgentStatus.SUCCESS.value,
        }

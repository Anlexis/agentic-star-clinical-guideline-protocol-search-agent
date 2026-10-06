"""AgentCore Platform v1.0"""

# HCR-C2-005 - InputValidateNode
# Domain node 1: parse and validate the incoming clinical-guideline search
# request.
#
# Caller data arrives on TWO channels, both validated here field-by-field:
#
#   input_context (structured invocation parameters, bridged into inner state -
#   see src/graph/context_bridge.py):
#     question         -> free-text clinical question (string)
#     category         -> guideline-category filter (inert identifier grammar)
#     top_k            -> maximum passages to keep (integer, 1-20)
#     score_threshold  -> relevance floor (number); may only be RAISED above
#                         the configured life-safety floor, never lowered
#
#   the string payload (validated_input, identifier-stripped by PreProcessNode
#   before it reaches the inner graph):
#     plain text                     -> the whole string is the clinical question
#     {"query"|"question", "category",
#      "top_k", "score_threshold"}   -> question + structured filters
#
# Validation is fail-CLOSED: a supplied field that is not the declared type, is
# non-finite (NaN / +-Infinity parse as floats but every comparison against them
# is False, which would silently disable the relevance floor), is out of range,
# or does not match the category grammar returns status=ERROR naming the FIELD -
# never echoing the rejected value. Absent caller data degrades to the
# plain-text path rather than fabricating an answer.
#
# Precedence: input_context values (already structured) win over the JSON
# envelope, which wins over the plain-text payload.
#
# The framework input gate masks PII and screens prompt injection on
# user_input / validated_input ONLY, so caller text arriving on the
# input_context channel would otherwise bypass both. This node therefore runs
# the SAME identifier strip and the SAME injection screen on
# input_context.question before it is used.
#
# execute(self, state) -> dict only - NO config parameter. Retrieval tuning
# reaches the node through State.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
import math
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import INPUT_REJECTED
from src.services.progress import emit_progress
from src.nodes.pre_process_node import _detect_prompt_injection, _strip_direct_identifiers
from src.schemas.state import to_json

# Hard cap on the normalised query length (defence-in-depth on input size; the
# server adapter additionally caps the whole serialized input_context).
_MAX_QUERY_CHARS = 2000

# Bounds for the caller-supplied top_k override.
_TOP_K_MIN = 1
_TOP_K_MAX = 20

# Bounds for the caller-supplied relevance floor. The configured floor (seeded
# into state from config/config.yaml) is the LOWER bound at runtime - a caller
# may tighten the floor, never relax it.
_SCORE_MIN = 0.0
_SCORE_MAX = 1.0

# Default relevance floor, mirroring config/config.yaml, used when state carries
# no seeded value (e.g. a unit test constructing the node directly).
_DEFAULT_SCORE_THRESHOLD = 0.75

# Guideline-category filter grammar: an inert lowercase identifier. The category
# is compared against knowledge-base metadata and reported in intake notes, so
# free text there would be caller-controlled content on an internal path.
_CATEGORY_RE = re.compile(r"^[a-z0-9_]{1,32}$")
_CATEGORY_FORMAT_MSG = "must be a lowercase identifier: letters, digits and underscores, 1-32 characters"

_WHITESPACE_RE = re.compile(r"\s+")

# Characters removed from the clinical question before it is stored. The
# question is echoed back in the rendered answer, so Markdown metacharacters
# and control characters (line breaks included) are stripped here - at the
# single point where caller text becomes state. The result is always a single
# line of inert text, so the question can no longer forge headings, list items,
# block quotes, horizontal rules, tables, code fences, links or a second
# disclaimer block in the report. Hyphens and parentheses are kept: clinical
# phrasing depends on them ("community-acquired", "72-year-old") and neither
# can open a structure once line breaks are gone.
_RENDER_UNSAFE_RE = re.compile(r"[\x00-\x1f\x7f`*_#\[\]<>|~\\]")


def _reject(field: str, reason: str, code: str = "INVALID_REQUEST") -> Dict[str, Any]:
    """Fail-closed rejection naming the offending FIELD only (no value echo)."""
    if code:
        # A value the caller can correct: the run COMPLETES carrying the
        # reason so the request can be sent again on the same conversation.
        emit_progress(INPUT_REJECTED)
        return {
            "status": AgentStatus.SUCCESS.value,
            "error_code": code,
            "error_log": [f"InputValidateNode: {field} {reason}"],
        }
    return {
        "status": AgentStatus.ERROR.value,
        "error_log": [f"InputValidateNode: {field} {reason}"],
    }


def _finite_in_range(
    raw: Any,
    field: str,
    minimum: float,
    maximum: float,
    *,
    integer: bool,
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """Parse a caller-supplied number, fail-closed.

    Rejects booleans (``isinstance(True, int)`` is True in Python), values that
    do not parse as a number, NaN / +-Infinity (they parse through ``float()``
    AND arrive via raw JSON, and every comparison against NaN is False - a NaN
    relevance floor would silently admit every passage), and values outside
    ``[minimum, maximum]``.

    Returns ``(value, None)`` on success and ``(None, error_dict)`` on
    rejection. ``raw is None`` means "not supplied on this channel" and is not
    an error.
    """
    if raw is None:
        return None, None
    if isinstance(raw, bool):
        return None, _reject(field, "must be a number, not a boolean")
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str) and raw.strip():
        try:
            value = float(raw.strip())
        except ValueError:
            return None, _reject(field, "must be a number")
    else:
        return None, _reject(field, "must be a number")
    if not math.isfinite(value):
        return None, _reject(field, "must be a finite number")
    if integer and value != int(value):
        return None, _reject(field, "must be a whole number")
    if value < minimum or value > maximum:
        return None, _reject(field, f"must be between {minimum} and {maximum}")
    return (float(int(value)) if integer else value), None


def _validate_category(raw: Any, field: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Validate a caller-supplied guideline category against the inert grammar."""
    if raw is None:
        return None, None
    if not isinstance(raw, str):
        return None, _reject(field, "must be a string")
    candidate = raw.strip().lower()
    if not candidate:
        return None, None
    if not _CATEGORY_RE.match(candidate):
        return None, _reject(field, _CATEGORY_FORMAT_MSG)
    return candidate, None


def _neutralise_for_render(text: str) -> str:
    """Collapse whitespace and drop characters that could forge report structure."""
    return _WHITESPACE_RE.sub(" ", _RENDER_UNSAFE_RE.sub(" ", text)).strip()


def _configured_score_floor(state: AgentState) -> float:
    """The configured relevance floor a caller may tighten but not relax."""
    seeded = state.get("retrieval_score_threshold")
    if isinstance(seeded, bool) or not isinstance(seeded, (int, float)):
        return _DEFAULT_SCORE_THRESHOLD
    value = float(seeded)
    if not math.isfinite(value):
        return _DEFAULT_SCORE_THRESHOLD
    return max(_SCORE_MIN, min(_SCORE_MAX, value))


class InputValidateNode(FunctionNode):
    """Validate the caller request into a normalised clinical question + filters.

    Input state keys:
        input_context:                caller invocation parameters (read-only)
        validated_input | user_input: identifier-stripped request payload
        retrieval_score_threshold:    configured relevance floor (scalar)

    Output state keys (partial dict):
        search_query:  normalised free-text clinical question
        query_filters: JSON dict {"category", "top_k", "score_threshold"}
        intake_notes:  (when anomalies were seen) JSON list[str]

    On a validation rejection (fail-closed): status=ERROR plus an error_log
    entry naming the field - the rest of the pipeline is skipped and no answer
    is produced.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")
        input_context = state.get("input_context", {}) or {}  # read-only
        if not isinstance(input_context, dict):
            return _reject("input_context", "must be an object")
        notes: List[str] = []

        # -- Structured caller data (input_context), field by field ------------
        ctx_question = input_context.get("question")
        if ctx_question is not None and not isinstance(ctx_question, str):
            return _reject("input_context.question", "must be a string")

        category, err = _validate_category(input_context.get("category"), "input_context.category")
        if err:
            return err

        top_k_value, err = _finite_in_range(
            input_context.get("top_k"), "input_context.top_k", _TOP_K_MIN, _TOP_K_MAX, integer=True
        )
        if err:
            return err
        top_k = int(top_k_value) if top_k_value is not None else None

        score_floor = _configured_score_floor(state)
        score_value, err = _finite_in_range(
            input_context.get("score_threshold"),
            "input_context.score_threshold",
            _SCORE_MIN,
            _SCORE_MAX,
            integer=False,
        )
        if err:
            return err
        if score_value is not None and score_value < score_floor:
            return _reject(
                "input_context.score_threshold",
                f"may only tighten the configured relevance floor of {score_floor}",
            )
        score_threshold = score_value

        query = ""
        if isinstance(ctx_question, str) and ctx_question.strip():
            # This channel bypasses both PreProcessNode and the framework input
            # gate, so the same identifier strip and injection screen run here.
            # The screen runs on the raw text AND on the render-safe form, so
            # normalisation can never assemble a payload that neither pass saw.
            candidate = _neutralise_for_render(ctx_question)
            for probe in (ctx_question, candidate):
                injection = _detect_prompt_injection(probe)
                if injection is None:
                    continue
                emit_trace_event(
                    "input_validate_injection_blocked",
                    {"pattern": injection, "field": "input_context.question"},
                    state,
                )
                return _reject(
                    "input_context.question",
                    f"rejected - prompt-injection pattern detected ({injection})",
                )
            # Neutralise BEFORE stripping so the class-labelled placeholders the
            # strip writes are not themselves stripped of their brackets.
            stripped, redacted_hits = _strip_direct_identifiers(candidate)
            if redacted_hits:
                notes.append("InputValidateNode: direct identifiers redacted from input_context.question.")
            query = stripped

        # -- String payload (plain text or JSON envelope) ----------------------
        if isinstance(raw, str) and raw.strip():
            payload: Any = None
            text = raw.strip()
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    notes.append(
                        "InputValidateNode: JSON-looking input did not parse - " "treated as plain text question."
                    )
            if isinstance(payload, dict):
                if not query:
                    query = _neutralise_for_render(str(payload.get("query") or payload.get("question") or ""))
                if category is None:
                    category, err = _validate_category(payload.get("category"), "category")
                    if err:
                        return err
                if top_k is None:
                    top_k_value, err = _finite_in_range(
                        payload.get("top_k"), "top_k", _TOP_K_MIN, _TOP_K_MAX, integer=True
                    )
                    if err:
                        return err
                    top_k = int(top_k_value) if top_k_value is not None else None
                if score_threshold is None:
                    score_value, err = _finite_in_range(
                        payload.get("score_threshold"),
                        "score_threshold",
                        _SCORE_MIN,
                        _SCORE_MAX,
                        integer=False,
                    )
                    if err:
                        return err
                    if score_value is not None and score_value < score_floor:
                        return _reject(
                            "score_threshold",
                            f"may only tighten the configured relevance floor of {score_floor}",
                        )
                    score_threshold = score_value
            elif not query:
                query = _neutralise_for_render(text)
        elif not query:
            notes.append("InputValidateNode: empty request - no clinical question to search.")

        # Every path above produced a render-safe, whitespace-collapsed query;
        # only the length cap remains.
        if len(query) > _MAX_QUERY_CHARS:
            query = query[:_MAX_QUERY_CHARS]
            notes.append(f"InputValidateNode: query truncated to {_MAX_QUERY_CHARS} chars.")

        filters: Dict[str, Any] = {
            "category": category,
            "top_k": top_k,
            "score_threshold": score_threshold,
        }

        # Domain audit: request parsed and normalised. Lengths and booleans
        # only - never the question text, the category or any caller value.
        emit_trace_event(
            "input_validate_complete",
            {
                "query_chars": len(query),
                "has_category_filter": category is not None,
                "has_top_k_override": top_k is not None,
                "has_score_threshold_override": score_threshold is not None,
            },
            state,
        )

        out: Dict[str, Any] = {
            "search_query": query,
            "query_filters": to_json(filters),
        }
        if notes:
            out["intake_notes"] = to_json(notes)
        return out

"""AgentCore Platform v1.0"""

# HCR-C2-005 - PreProcessNode (outer pre_process slot: trust gate, identifier
# screen, prompt-injection screen, domain audit)
#
# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus.<X>.value strings for status assignments
#  - Read input_context via state.get("input_context", {}) — read-only
#  - Never import from mediator/, api/, or other agents
#
# Trust gate: this is a life-safety healthcare domain — the outer pre_process
# slot requires VERIFIED_EXTERNAL (not the framework-implicit ANONYMOUS
# default), and rejects empty / non-string input before the inner clinical
# workflow runs.
#
# Identifier + injection screen (see docs/02_design.md, "Security Gates"): this
# node surface-strips direct identifiers out of the clinical question BEFORE it
# reaches the inner workflow, and screens the same text for prompt-injection
# attempts against the clinical safety contract. Both run INLINE in execute()
# through module-level helpers: the framework auto-wraps an
# `_extra_security_gate_input` INSTANCE METHOD into the graph chain and then
# passes None downstream, so that hook is not used on a node class here.
#
# Layering — the framework runs FIRST, this node runs SECOND:
# `FunctionNode._security_gate_input()` executes inside `BaseNode.__call__()`
# before execute() and masks e-mail, phone (JP/US), US SSN, credit-card and My
# Number digit groups, and Latin Title-Case name bigrams in `user_input` /
# `validated_input` to the generic marker `[MASKED]`. The screen below is the
# DOMAIN layer stacked on top of it. It covers the direct identifiers a Japanese
# clinical record carries that the framework default does not model —
# label-anchored patient / medical-record numbers, dates of birth, postal and
# Japanese street addresses, and Japanese personal names — and it re-covers the
# framework's own classes as defence in depth, so the strip still holds where
# the framework gate does not run: this helper is also applied by
# InputValidateNode to caller text arriving on the input_context channel, which
# the framework gate never inspects. Redaction is to a stable,
# class-labelled placeholder (`[REDACTED:PHONE]`, …) that keeps the clinical
# sentence readable; the raw value is never logged, never audited, and never
# written to State.
#
# False-positive discipline (life-safety, deliberate): every numeric rule is
# either LABEL-ANCHORED (a patient / MRN / DOB / address keyword must introduce
# the value) or shape-specific enough that clinical prose cannot collide with
# it. Bare numbers are NEVER redacted, so drug doses ("500 mg", "1.25 mg/kg"),
# ICD-10 codes ("J18.9"), protocol identifiers ("protocol 7-2"), guideline
# revision dates ("revised 2024-06-01") and ages ("72-year-old") pass through
# untouched. Over-redacting a dose or a code would be a worse clinical defect
# than the gap this screen closes, so precision is preferred to recall on every
# rule that could touch clinical numerals.

import re
from typing import Any, Callable, ClassVar, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT

# ---------------------------------------------------------------------------
# Direct-identifier patterns
# ---------------------------------------------------------------------------
# Each rule is (class_name, compiled_pattern, skip_guard). When a pattern
# defines the named group `v`, ONLY that span is replaced — the surrounding
# label ("patient id:", "生年月日") is preserved so the redaction is auditable
# and the clinical sentence stays readable. `skip_guard` (optional) receives the
# captured value and returns True to leave the match alone.

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Label-anchored ONLY: a bare digit run is never a patient id (it is a dose, a
# protocol number, a lab value or an age). At least 4 digits, optional short
# alpha prefix and hyphenated suffix (e.g. "MRN A-4451233", "患者ID 88231145").
_PATIENT_ID_RE = re.compile(
    r"(?:patient\s*(?:ids?|no\.?|number|#)"
    r"|pt\.?\s*(?:id|no\.?)"
    r"|mrn"
    r"|medical\s+record\s*(?:no\.?|number|#)"
    r"|chart\s*(?:no\.?|number|#)"
    r"|患者\s*ID|患者番号|患者識別番号|カルテ番号|診察券番号)"
    r"\s*[:：#=]?\s*"
    r"(?P<v>[A-Za-z]{0,4}-?\d{4,}(?:-[A-Za-z0-9]+)*)",
    re.IGNORECASE,
)

# Label-anchored ONLY. An unlabelled ISO date is far more likely to be a
# guideline revision / review date than a birth date, and redacting those would
# corrupt the citation trail.
_DOB_RE = re.compile(
    r"(?:\bdate\s+of\s+birth\b|\bbirth\s*date\b|\bdob\b|\bd\.o\.b\.?|\bborn\s+on\b"
    r"|生年月日|誕生日)"
    r"\s*[:：=]?\s*"
    r"(?P<v>\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*日?"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{4})",
    re.IGNORECASE,
)

# Japanese postal code — the 〒 mark makes this unambiguous.
_JP_POSTCODE_RE = re.compile(r"〒\s*(?P<v>\d{3}\s*-?\s*\d{4})")

# Label-anchored address. The English label REQUIRES a separator: "address" is
# an ordinary clinical verb ("how do we address hyperkalemia") and must not
# trigger on its own. The value stops at sentence punctuation so a redaction can
# never swallow the clinical question that follows it.
_ADDRESS_LABEL_RE = re.compile(
    r"(?:\baddress\b\s*[:：=]|(?:現)?住所\s*[:：=]?)\s*(?P<v>[^\n。、.?？!！]{2,60})",
    re.IGNORECASE,
)

# Japanese street address: 都道府県 → 市区町村 → hyphenated banchi digits.
# All three parts are required, which clinical guidance text never satisfies.
_JP_STREET_RE = re.compile(
    r"(?P<v>[一-龥ぁ-んァ-ヶー]{2,10}?[都道府県]" r"[^\s\n]{1,15}?[市区町村]" r"[^\s\n]{0,20}?\d+(?:[-−]\d+){1,3})"
)

# My Number / 12-digit run (mirrors the framework shape, with the same guards
# against slicing a longer digit sequence such as a credit-card number).
_MY_NUMBER_RE = re.compile(r"(?<!\d[-\s])\b(?P<v>\d{4}[-\s]?\d{4}[-\s]?\d{4})(?![-\s]?\d)\b")

# National-identifier shape (3-2-4 digit groups). The framework masks this
# class on the string payload; re-covering it here means the SAME redaction
# applies wherever this helper is called, including the caller-context channel
# the framework gate never sees. A clinical date is 4-2-2, a dose is not
# hyphen-grouped, so the shape does not collide with clinical numerals.
_NATIONAL_ID_RE = re.compile(r"(?<!\d)(?P<v>\d{3}-\d{2}-\d{4})(?!\d)")

_PHONE_JP_RE = re.compile(r"(?<!\d)(?P<v>\+81[-\s]?\d{1,4}[-\s]\d{1,4}[-\s]\d{4}" r"|0\d{1,4}[-\s]\d{1,4}[-\s]\d{4})\b")

_PHONE_US_RE = re.compile(r"(?<!\d)(?P<v>(?:\+1[-.\s]?)?(?:\(\d{3}\)[-.\s]?|\d{3}[-.\s])\d{3}[-.\s]\d{4})\b")

# Personal name, label-anchored. The label must contain "patient" (or the
# Japanese equivalent) and be followed by a separator — a bare "name:" is
# rejected on purpose because clinical text says "drug name: amoxicillin".
_NAME_LABEL_EN_RE = re.compile(
    r"(?:\bpatient(?:\s*name)?|\bpt\.?\s*name)\s*[:：]\s*"
    r"(?P<v>[A-Z][A-Za-z'\-]{1,20}(?:\s+[A-Z][A-Za-z'\-]{1,20}){0,3})"
)

_NAME_LABEL_JA_RE = re.compile(
    r"(?:患者氏名|患者名|氏名|お名前)\s*[:：]?\s*" r"(?P<v>[一-龥ァ-ヶ][一-龥ぁ-んァ-ヶー]{1,7})"
)

# Honorific form (山田太郎さん). Guarded below, because "患者さん" /
# "高齢者さん" are ordinary Japanese clinical phrasing, not personal names.
_NAME_HONORIFIC_JA_RE = re.compile(r"(?P<v>[一-龥ァ-ヶ][一-龥ぁ-んァ-ヶー]{1,7})(?=\s*(?:さん|様))")

# Generic clinical nouns that legitimately precede さん / 様. Matched as a
# SUFFIX so compounds ("成人患者さん", "入院患者さん") are covered too.
_GENERIC_HONORIFIC_NOUNS: Tuple[str, ...] = (
    "患者",
    "利用者",
    "高齢者",
    "対象者",
    "本人",
    "家族",
    "医師",
    "看護師",
    "薬剤師",
    "担当者",
    "主治医",
    "保護者",
    "被験者",
    "妊婦",
    "小児",
    "成人",
    "皆",
    "皆様",
    "御家族",
)


def _is_generic_honorific(value: str) -> bool:
    """True when a さん/様 match is a common clinical noun, not a personal name."""
    return value.endswith(_GENERIC_HONORIFIC_NOUNS)


# Order matters: label-anchored rules run before the shape-only rules so the
# more specific class wins the audit label.
_DIRECT_IDENTIFIER_RULES = [
    ("EMAIL", _EMAIL_RE, None),
    ("PATIENT_ID", _PATIENT_ID_RE, None),
    ("DOB", _DOB_RE, None),
    ("ADDRESS", _JP_POSTCODE_RE, None),
    ("ADDRESS", _ADDRESS_LABEL_RE, None),
    ("ADDRESS", _JP_STREET_RE, None),
    ("MY_NUMBER", _MY_NUMBER_RE, None),
    ("NATIONAL_ID", _NATIONAL_ID_RE, None),
    ("PHONE", _PHONE_JP_RE, None),
    ("PHONE", _PHONE_US_RE, None),
    ("NAME", _NAME_LABEL_EN_RE, None),
    ("NAME", _NAME_LABEL_JA_RE, None),
    ("NAME", _NAME_HONORIFIC_JA_RE, _is_generic_honorific),
]


def _strip_direct_identifiers(text: str) -> Tuple[str, List[str]]:
    """Surface strip: redact direct identifiers to class-labelled placeholders.

    Returns ``(redacted_text, hits)`` where ``hits`` holds one class name per
    redaction performed, so the caller can report both a count and the distinct
    classes. The raw identifier values are NEVER returned, logged or audited.
    """
    hits: List[str] = []
    redacted = text

    for class_name, pattern, guard in _DIRECT_IDENTIFIER_RULES:
        placeholder = f"[REDACTED:{class_name}]"

        def _replace(
            match: "re.Match[str]",
            _name: str = class_name,
            _guard: Optional[Callable[[str], bool]] = guard,
            _ph: str = placeholder,
        ) -> str:
            has_value_group = "v" in match.groupdict() and match.group("v") is not None
            value = match.group("v") if has_value_group else match.group(0)
            if _guard is not None and _guard(value):
                return match.group(0)  # legitimate clinical phrasing — leave it
            hits.append(_name)
            if not has_value_group:
                return _ph
            whole = match.group(0)
            start = match.start("v") - match.start(0)
            end = match.end("v") - match.start(0)
            return whole[:start] + _ph + whole[end:]

        redacted = pattern.sub(_replace, redacted)

    return redacted, hits


# ---------------------------------------------------------------------------
# Prompt-injection screen
# ---------------------------------------------------------------------------
# Every pattern requires an object that unambiguously refers to THIS agent's own
# instruction context ("instructions", "prompt", "disclaimer", …). A bare
# directive verb is never enough: a clinician legitimately asks "when can we
# disregard the guideline recommendation" or "which drugs act as a substrate",
# and a life-safety template must not refuse those. A match is REJECTED rather
# than sanitised — an input trying to subvert the grounding / disclaimer
# contract has no safe interpretation in this domain.

_INJECTION_RULES: List[Tuple[str, "re.Pattern[str]"]] = [
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
            r"\b(?:previous|prior|preceding|above|earlier|initial|original|system|all)\s+"
            r"(?:instructions?|prompts?|directives?)\b"
            r"|\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
            r"\byour\s+(?:instructions?|prompt|rules?|constraints?|guardrails?)\b"
            r"|(?:指示|命令|プロンプト)\s*(?:を|は)\s*(?:無視|忘れ|破棄|上書き)",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_probe",
        re.compile(
            r"\b(?:reveal|show|display|print|repeat|output|dump|tell\s+me|give\s+me)\b"
            r"[^.\n]{0,30}?\b(?:your|the)\s+(?:system\s+)?(?:prompt|instructions|configuration)\b"
            r"|\bsystem\s+prompt\b"
            r"|システム\s*プロンプト"
            r"|(?:指示|プロンプト)(?:文|内容)?\s*を\s*(?:表示|出力|教え)",
            re.IGNORECASE,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"\byou\s+are\s+now\b"
            r"|\bfrom\s+now\s+on\b[^.\n]{0,30}\byou\b"
            r"|\bpretend\s+to\s+be\b"
            r"|\bdeveloper\s+mode\b"
            r"|\bjailbreak\b"
            r"|あなたは(?:今|これ)から"
            r"|のふりをして",
            re.IGNORECASE,
        ),
    ),
    (
        "guardrail_bypass",
        re.compile(
            r"\b(?:without|omit|remove|suppress|drop|skip|no)\s+(?:the\s+)?"
            r"(?:advisory\s+)?disclaimer\b"
            r"|\bdo\s+not\s+(?:include|add|append|show)\s+(?:the\s+)?(?:advisory\s+)?disclaimer\b"
            r"|(?:免責|注意)(?:事項|文言|文)?\s*を?\s*(?:省略|削除|外して|付けないで|表示しないで)",
            re.IGNORECASE,
        ),
    ),
]


def _detect_prompt_injection(text: str) -> Optional[str]:
    """Return the name of the first prompt-injection pattern matched, else None."""
    for name, pattern in _INJECTION_RULES:
        if pattern.search(text):
            return name
    return None


class PreProcessNode(FunctionNode):
    """Validate, screen and enrich incoming input before main processing."""

    # Explicit by design, not inherited implicitly. Life-safety healthcare
    # domain — the outer gate slot requires VERIFIED_EXTERNAL.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> dict[str, Any]:
        user_input = state.get("user_input", "")
        input_context = state.get("input_context", {})  # read-only

        # Non-empty string only.
        if not isinstance(user_input, str) or not user_input.strip():
            emit_progress(EMPTY_INPUT)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "EMPTY_INPUT",
                "error_log": ["PreProcessNode: user_input is empty or missing"],
            }

        candidate = user_input.strip()

        # (a) Prompt-injection screen — reject before anything downstream sees
        # the payload.
        injection = _detect_prompt_injection(candidate)
        if injection is not None:
            # Audit the block by PATTERN NAME only. The rejected text is never
            # emitted, never logged and never written to State.
            emit_trace_event(
                "pre_process_injection_blocked",
                {"pattern": injection, "input_chars": len(candidate)},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PreProcessNode: input rejected - prompt-injection " f"pattern detected ({injection})"],
            }

        # (b) Surface-strip direct identifiers before the inner workflow.
        validated_input, redacted_hits = _strip_direct_identifiers(candidate)
        redacted_classes = sorted(set(redacted_hits))

        # Domain audit: record that a clinical query was accepted. Counts and
        # CLASS NAMES only — no redacted value ever enters the payload.
        emit_trace_event(
            "pre_process_complete",
            {
                "input_chars": len(validated_input),
                "identifiers_redacted": len(redacted_hits),
                "identifier_classes": redacted_classes,
            },
            state,
        )

        return {
            "validated_input": validated_input,
            "enriched_context": {
                "source": "ClinicalGuidelinesQAAgent",
                "channel": input_context.get("channel", "unknown"),
            },
            "status": AgentStatus.SUCCESS.value,
        }

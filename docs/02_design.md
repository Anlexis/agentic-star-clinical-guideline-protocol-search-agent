# Template Design Specification — HCR-C2-005

**Template ID:** HCR-C2-005
**Template Name:** ClinicalGuidelinesQAAgent
**Category:** Cat 2 (multi-step domain workflow — RAG pattern)
**Industry:** HCR

## Position in AgentCore Architecture

| Field | Value |
|---|---|
| Agent Class | `ClinicalGuidelinesQAAgent` (alias `Graph`) |
| L1 Base (framework base class) | AgentBaseGraph — direct framework inheritance |
| Inner graph base | `BaseGraph` — `DomainWorkflowGraph` |
| Pattern | Cat 2 two-layer nested architecture (outer fixed 5-node backbone + `GraphNode` in the `main` slot wrapping an inner `BaseGraph` domain workflow) |

- **Separation of concerns:**
  - State: flat `TypedDict` composition (no Pydantic — msgpack incompatible);
    structured fields stored as JSON strings via `to_json()` / `from_json()`;
    retrieval tuning stored as plain SCALAR fields (see "Config forwarding" below)
  - Node: framework inheritance via `FunctionNode` (override
    `execute(self, state) -> dict` only — **no `config` parameter**)
  - Graph: composition (`register_nodes()` for node substitution); outer
    `add_edges()` is NOT overridden

## Purpose

Clinical Guidelines & Protocol Knowledge Base Search Agent: a clinician
(physician, pharmacist, or quality/stewardship-team member) submits a
natural-language clinical question — treatment protocols, dosage references,
contraindication checks — and the agent retrieves the most relevant
guideline/protocol passages and generates a grounded answer with numbered
citations, or an explicit insufficient-evidence decline when no passage
clears the relevance floor. Life/safety-adjacent domain: retrieval is tuned
to the healthcare relevance floor (`score_threshold: 0.75`) and generation is fully
deterministic (`temperature: 0.0`) so answers are strictly grounded and never
fall back on unsourced parametric knowledge. Every answer — including the
decline — carries a mandatory, non-suppressible advisory disclaimer. v1 is
fully deterministic (keyword retrieval + rule-based grounded answer assembly;
no live LLM call — see the v1 Implementation Note below). Scope is advisory
guideline retrieval only: no prescribing, ordering, dosing calculation,
diagnosis, or EHR write-back, and no patient-record/PHI ingestion (a
published-guideline reference KB, not a patient-data system).

## Architecture Overview

### Outer backbone (AgentBaseGraph)

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                     ↓ (retry, max 3)
                                   pre_process
```

| Slot | Class | Responsibility | required_trust_level | Gate |
|------|-------|----------------|----------------------|------|
| initialize | InitializeNode (framework default) | session_id, trust_level, schema_version | — (framework) | — |
| pre_process | `PreProcessNode` | validate non-empty input (empty / non-string ends the run by completing with a correction message — see "Rejection Contract"); surface-strip direct identifiers; prompt-injection screen (terminates the run) → `validated_input` | `TrustLevel.VERIFIED_EXTERNAL` | input |
| main | `ClinicalGuidelineSearchGraphNode` (`GraphNode`) | delegates to inner `DomainWorkflowGraph`; stashes `input_context` on the context bridge; maps inner `formatted_answer` → outer `result`; skips the inner workflow entirely when a reason was already settled upstream | — (GraphNode delegation) | — (delegates) |
| post_process | `PostProcessNode` | output gate — module-level `_security_gate_output()` scan + advisory-disclaimer verification + citation-grounding check → ERROR + sanitised stub; on a run declined upstream it renders the caller-facing sentence instead of gating an answer | `TrustLevel.VERIFIED_EXTERNAL` | output |
| finalize | FinalizeNode (framework default) | response_metadata, total_time_ms | — (framework) | — |

### Inner graph (DomainWorkflowGraph — BaseGraph, linear)

```
START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

All five inner domain nodes declare `required_trust_level = TrustLevel.ANONYMOUS`
(the external trust gate lives on the outer backbone gate node; a stricter inner
level would deny a real VERIFIED_EXTERNAL invoke at runtime).

| Node | Responsibility | required_trust_level | Input State | Output State |
|------|----------------|----------------------|-------------|--------------|
| `InputValidateNode` | Validate the caller-data contract on BOTH channels (`input_context` + the string payload): finite/bounded numerics, the inert `category` grammar, the tighten-only relevance floor; identifier-strip and injection-screen context-channel text; normalise the question to a single inert line and cap its length | `TrustLevel.ANONYMOUS` | `input_context`, `validated_input` \| `user_input` | `search_query`, `query_filters`, `intake_notes` |
| `RetrieveNode` | Deterministic keyword retrieval over the seeded KB (`config/kb/hcr_clinical_guideline_kb.json`): tokenise query, score title/tags/content overlap, apply category filter | `TrustLevel.ANONYMOUS` | `search_query`, `query_filters`, `retrieval_top_k`, `retrieval_kb_path` | `retrieved_documents`, `intake_notes` |
| `RerankFilterNode` | Rerank candidates (category-match boost), drop entries below `score_threshold` (healthcare floor 0.75, or a stricter caller value), cap at `top_k` — this is the mechanism behind the "decline on insufficient confidence" requirement | `TrustLevel.ANONYMOUS` | `retrieved_documents`, `query_filters`, `retrieval_top_k`, `retrieval_score_threshold` | `ranked_documents` |
| `GenerateAnswerNode` | Rule-based grounded answer assembly from the ranked KB passages only, with numbered citation markers; explicit insufficient-evidence decline when nothing survived the relevance floor (v1 deterministic — LLM synthesis seam documented below) | `TrustLevel.ANONYMOUS` | `ranked_documents`, `search_query` | `grounded_answer`, `citations` |
| `OutputFormatNode` | Compose the final answer: body + Sources list + the mandatory, non-configurable advisory disclaimer (this node composes it; the output gate independently verifies it) | `TrustLevel.ANONYMOUS` | `grounded_answer`, `citations` | `formatted_answer`, `status` |

### Data Flow

```
user_input + input_context
  → PreProcessNode                                 → validated_input
  → ClinicalGuidelineSearchGraphNode.extract_input → context bridge stashes input_context
                                                   → inner DomainWorkflowGraph.invoke(validated_input)
        → _extra_initial_state()                   → seeds input_context + retrieval scalars
        → input_validate                           → search_query / query_filters
        → retrieve                                 → retrieved_documents
        → rerank_filter                            → ranked_documents
        → generate_answer                          → grounded_answer / citations (or insufficient-evidence decline)
        → output_format                            → formatted_answer (+ advisory disclaimer)
     get_output() → {formatted_answer, citations, status, error_code, ...}
  → ClinicalGuidelineSearchGraphNode.merge_output  → result = formatted_answer, guideline_answer
  → PostProcessNode (output gate)                  → formatted_output (gated)
```

A request declined on a caller-correctable value takes the same path with the
work skipped — the marker is the only thing that travels:

```
user_input + input_context
  → PreProcessNode | InputValidateNode              → error_code set, no answer produced
  → every node still ahead                          → passes error_code through, does nothing
  → PostProcessNode                                 → formatted_output = the caller-facing sentence
```

### Caller-data contract

Structured parameters reach the pipeline on either of two channels, and
`InputValidateNode` validates both field by field:

| Field | Channel | Rule |
|---|---|---|
| `question` | `input_context` | string; identifier-stripped and injection-screened here (the framework input gate covers `user_input` / `validated_input` only, so this channel would otherwise bypass both); normalised to a single inert line and capped at 2000 chars |
| `category` | `input_context` or JSON envelope | must match `[a-z0-9_]{1,32}` — an inert identifier, because it reaches knowledge-base comparisons and intake notes |
| `top_k` | `input_context` or JSON envelope | finite whole number in 1–20 |
| `score_threshold` | `input_context` or JSON envelope | finite number in 0.0–1.0 AND not below the configured relevance floor — a caller may tighten it, never relax it |

Every numeric goes through one finite + bounded parser. `NaN` / `±Infinity`
parse through `float()` and arrive through raw JSON, and every comparison
against `NaN` is False — a `NaN` relevance floor would silently admit every
passage, which is the exact decision this template exists to make. Validation
therefore fails **closed**: nothing is retrieved, no answer is produced, and
the offending FIELD is named in `error_log` without the value ever being
echoed. Fail-closed here means the work does not happen, not that the run
terminates — a value the caller can correct ends the run by completing with a
reason marker and a correction sentence, so the request can be fixed and sent
again. The full contract is "Rejection Contract" below.

`GraphNode.execute()` does not forward `input_context` to the inner graph, so
`src/graph/context_bridge.py` carries it across the boundary: `extract_input()`
stashes it in a `ContextVar` immediately before the inner invoke, and the inner
graph's `_extra_initial_state()` reads it back while building the initial
state. A `ContextVar` keeps concurrent invocations in one process isolated.

The string payload still accepts a JSON envelope (`{"query": ..., "category":
..., "top_k": ...}`): it passes through `validated_input` as a string and the
first inner node parses it back. `input_context` values win over the envelope,
which wins over plain text.

### Config forwarding (`_parent_config()` → State, not per-call `execute()`)

> **Node contract:** every `FunctionNode` implements EXACTLY
> `def execute(self, state) -> dict` — no extra parameters (`config` etc.),
> and no `__init__` / ctor args. Configuration reaches a node through State.

`config/agent.yaml` is the static manifest; the runtime parameters live in
`config/config.yaml` and are read by `load_runtime_config()`.
`src/api/server.py` passes that mapping into the agent constructor, so the
declared `max_retry` is live in the backbone's retry routing, and
`ClinicalGuidelineSearchGraphNode._parent_config()` forwards the tuning blocks
to the inner graph under `config["configurable"]` (never `{}`):

```
{"configurable": {"retrieval": {top_k, score_threshold, kb_path},
                  "llm": {...},
                  "max_retry": N, "timeout_seconds": N}}
```

`timeout_s` is renamed to `timeout_seconds` on the way through — the key the
inner graph validates. Both runtime numbers are validated when present
(`_validate_config()` on each graph layer), so a declared-but-broken value
fails loudly at compile time instead of degrading silently.

`get_subgraph()` passes this into `DomainWorkflowGraph(config=...)`. The
inner graph's constructor (`BaseGraph.__init__(self, config=None)`) is a
graph-level constructor, not a node — accepting `config` there does not
violate the node `execute()` contract. `DomainWorkflowGraph._extra_initial_state()`
then republishes the `retrieval` block into the inner initial state as three
plain SCALAR fields — `retrieval_top_k`, `retrieval_score_threshold`,
`retrieval_kb_path` — so the declared values are live at runtime *through
State*, never through a node method parameter. `RetrieveNode` and
`RerankFilterNode` read these fields directly off `state` (`state.get(
"retrieval_top_k")`, etc.), falling back to module defaults that mirror
`config/config.yaml` when a field is unseeded (e.g. a unit test that
constructs a node directly and calls `execute()` with a minimal state).

### State Definition

| Field | Type | Purpose | Layer |
|-------|------|---------|-------|
| `validated_input` | `Optional[str]` | identifier-stripped clinical question | outer |
| `guideline_answer` | `Optional[str]` | final answer, mapped from inner `formatted_answer` | outer |
| `search_query` | `Optional[str]` | normalised clinical question | inner |
| `query_filters` | `Optional[str]` (JSON) | validated structured params (`category`, `top_k`, `score_threshold`) | inner |
| `retrieval_top_k` | `Optional[int]` | state-seeded runtime value (scalar) | inner |
| `retrieval_score_threshold` | `Optional[float]` | state-seeded runtime value (scalar) | inner |
| `retrieval_kb_path` | `Optional[str]` | state-seeded runtime value (scalar) | inner |
| `retrieved_documents` | `Optional[str]` (JSON) | scored KB candidates | inner |
| `ranked_documents` | `Optional[str]` (JSON) | reranked + threshold-filtered passages | inner |
| `grounded_answer` | `Optional[str]` | rule-assembled grounded answer body, or the decline | inner |
| `citations` | `Optional[str]` (JSON) | `[{ref, id, title, source}]` | inner |
| `formatted_answer` | `Optional[str]` | final answer + sources + advisory disclaimer | inner |
| `intake_notes` | `Optional[str]` (JSON) | validation / parse notes (no patient data) | inner |
| `error_code` | `Optional[str]` | reason marker for a request declined on a caller-correctable value; set by the declining node, passed through untouched by every node after it, and resolved to the caller-facing sentence by `PostProcessNode` — see "Rejection Contract" | both |
| `trace_id` / `correlation_id` | `Optional[str]` | framework-managed tracing | both |

**State Constraints (mandatory):**
- Flat `TypedDict` only (primitives + JSON-serialisable types).
- Structured fields (dict / list[dict]) stored as JSON STRINGS via `to_json()` /
  `from_json()` — used consistently by every producer AND consumer (msgpack
  safety). Retrieval tuning is stored as plain scalars (int/float/str), not
  JSON — it is never a dict/list State value.
- Domain fields are `Optional[...]` (valid TypedDict before any node writes).
- `formatted_output` is NOT re-declared (backbone field stays framework-owned).
- No JWT, API keys, credentials, or raw patient identifiers in State.
- `InvocationContext` via `config["configurable"]` only (never in State).
- No Pydantic models / dataclasses / arbitrary Python objects.

## Rejection Contract

A refused request has two possible endings, and which one applies turns on a
single question: can the caller fix the request and send it again?

### A caller-correctable value ends the run by COMPLETING

The node that finds the problem returns `status: AgentStatus.SUCCESS.value`
together with an `error_code` marker, and the run carries that marker to the
end of the pipeline instead of terminating. Terminating would end the calling
surface's conversation turn and surface only a status, leaving the reason
reachable solely from the audit trail; completing lets the caller correct the
value and resend on the same turn.

| Condition | Node | `error_code` |
|---|---|---|
| `user_input` absent, not a string, or empty / whitespace-only | `PreProcessNode` | `EMPTY_INPUT` |
| any caller-data validation failure on either channel — `input_context` not an object, `question` not a string, the inert `category` grammar, a non-finite / out-of-range / boolean / fractional / non-numeric `top_k` or `score_threshold`, or a `score_threshold` below the configured relevance floor | `InputValidateNode` | `INVALID_REQUEST` |
| prompt injection detected on `input_context.question` | `InputValidateNode` | `INVALID_REQUEST` |

Once the marker is set, every node still ahead of the run passes it through
untouched and does no work of its own: the outer `main` `GraphNode` skips the
inner workflow entirely, and each inner node (`retrieve`, `rerank_filter`,
`generate_answer`, `output_format`) returns the marker unchanged. A reason
settled early is the real one — continuing to run the pipeline over input that
was already declined would produce a second, vaguer reason and overwrite the
specific one.

`PostProcessNode` turns the marker into the one plain sentence the caller reads
(`src/services/failure_message.py`) and emits `post_process_degraded`. Each
sentence names WHAT to correct and nothing else: it never echoes the rejected
value, names an internal field path, or quotes a gate message — those stay in
`error_log`, the internal audit channel. A marker with no mapped sentence falls
back to the generic one rather than surfacing the code itself.

`error_code` is deliberately NOT part of the envelope the outer graph returns
(`AgentBaseGraph.get_output()` surfaces `output`, `status`, `trace_id`,
`correlation_id`, `node_history`). The marker is an internal routing field; the
reason reaches the caller through the message text alone, which keeps the
caller-facing contract to one stable sentence instead of a code vocabulary that
downstream callers would start branching on.

### A refusal the agent makes on its own behalf TERMINATES

These endings are not caller-correctable and do **not** complete: the run ends
with `status: AgentStatus.ERROR.value`.

| Condition | Where |
|---|---|
| prompt injection detected on the string payload — instruction override, system-prompt probe, role override, disclaimer / guardrail bypass | `PreProcessNode` |
| output gate violation — disallowed content, missing advisory disclaimer, or a citation marker not listed in the Sources section | `PostProcessNode` |
| caller trust level below a node's `required_trust_level` (S-1) | framework gate |
| high-confidence prompt injection on `user_input` / `validated_input` (S-2), or a credential pattern found in a node's result (S-3) | framework gates |
| any exception raised inside a node's `execute()` — including the `SubgraphError` the `propagate` error strategy re-raises out of the inner graph, and a State field that fails JSON serialisation | framework `BaseNode.__call__()` |

An ERROR in State is terminal for the remainder of the run: `BaseNode.__call__()`
skips the `execute()` of every node after it, and the backbone `route()` sends
the run straight to `finalize`, so `post_process` never runs and the returned
`output` is empty. That is the intended shape — a refusal the caller cannot act
on must not be dressed up as an answer, and an input trying to subvert the
grounding or the mandatory disclaimer has no safe interpretation in a
life-safety domain.

## Security Gates

- **Trust gate / input validation:** every node declares
  `required_trust_level` (see tables above); `PreProcessNode`
  (VERIFIED_EXTERNAL) refuses empty / non-string `user_input` before the inner
  workflow runs — the run completes carrying `error_code: EMPTY_INPUT` and the
  caller-facing correction sentence, and nothing is retrieved or answered. A
  caller below VERIFIED_EXTERNAL is denied by the S-1 gate instead, which
  terminates the run with `AgentStatus.ERROR` before `execute()` is reached.
  The standalone server elevates authenticated Bearer callers to
  VERIFIED_EXTERNAL (`INVOKE_AUTH_TOKEN`).
- **Identifier screen:** `PreProcessNode` surface-strips direct
  identifiers from the payload; the framework `FunctionNode` default PII scan
  additionally masks `user_input` / `validated_input` at every node boundary.
  This template does not ingest patient records — only the normalised
  question, KB passage summaries, and the final answer are persisted.
  The strip runs INLINE in `execute()` via module-level helpers (no
  `_extra_security_gate_input` instance method — see below), covers **EMAIL,
  PATIENT_ID (patient / MRN / chart / カルテ番号), DOB, ADDRESS (〒postal,
  labelled, Japanese street), MY_NUMBER, NATIONAL_ID (3-2-4 digit groups),
  PHONE (JP/US) and NAME (labelled EN/JA + Japanese honorific)**, and replaces
  each hit with a stable class-labelled placeholder (`[REDACTED:PATIENT_ID]`, …)
  so the clinical sentence stays readable and the redaction stays auditable.
  Redacted values are never logged, audited or written to State. The same node
  also screens the payload for **prompt injection** against the clinical safety
  contract (instruction override, system-prompt probe, role override,
  disclaimer/guardrail bypass; EN + JA) and REJECTS a match with
  `AgentStatus.ERROR` — an input trying to subvert the grounding or the
  mandatory disclaimer has no safe interpretation in a life-safety domain.
  Both screens are precision-first: every numeric rule is label-anchored or
  shape-specific, so drug doses, ICD-10 codes, protocol identifiers, guideline
  revision dates and ages are never redacted, and a clinician's legitimate
  "when can we disregard the guideline recommendation" is never refused.
  **These are the node's OWN guarantees, not the framework's**: the same
  helpers are re-applied by `InputValidateNode` to caller text arriving on the
  `input_context` channel, which the framework gate never inspects, and the
  test suite drives both screens through `execute()` directly — with no
  framework wrapper in front — so the refusal cannot silently depend on a layer
  above. The two channels end the run differently: on the string payload the
  screen terminates with `AgentStatus.ERROR`, while on the `input_context`
  channel the same match is one of the field-level rejections
  `InputValidateNode` owns, so it completes with `error_code: INVALID_REQUEST`
  and the correction sentence. The rejected text is discarded identically
  either way — it never becomes `search_query`, never reaches retrieval, and is
  never echoed back to the caller.
- **Output gate:** `PostProcessNode` calls the module-level
  `_security_gate_output()` scan from `execute()` and enforces three
  invariants, in this order:
  1. **Disallowed content** — API keys, JWTs, Bearer tokens, credential
     assignments, medical-record-number and national-identifier shapes. The
     scan is RECURSIVE, so a violation nested inside a structured result is
     caught exactly like a top-level string. A hit replaces the output with a
     sanitised stub and returns `AgentStatus.ERROR`.
  2. **Mandatory advisory disclaimer** — the answer must carry the advisory
     line `OutputFormatNode` composes. The gate does not compose it; it
     verifies it, so a future pipeline change that drops the disclaimer blocks
     the answer instead of shipping it bare.
  3. **Citation grounding** — every numbered citation marker in the answer body
     must resolve to an entry in the rendered Sources list.

  All three checks run only when the pipeline actually produced an answer. On a
  run declined upstream there is nothing to gate: the node renders the
  caller-facing sentence for the carried `error_code` and completes.

  The pattern scan runs FIRST, on the untouched result: a transformation that
  rewrites characters inside a matched span would destroy the very shape the
  scan keys on, so nothing may be inserted before it. This template renders no
  monetary or statistical aggregates — the answer body is knowledge-base
  passage text plus the caller's own question — so there is no rounding or
  precision invariant to enforce, and clinical numerals (doses, ICD-10 codes,
  protocol numbers, revision dates) are reproduced verbatim by design.

  No `_extra_security_gate_input` / `_extra_security_gate_output` instance
  methods are defined on any node (the framework auto-wraps such hooks).
- **Audit logging:** every node's `execute()` emits exactly ONE
  domain-specific `emit_trace_event("<node>_complete", {small payload carrying
  no patient data}, state)` (free function, positional args) on its success
  path. Nodes do NOT emit `node_start` / `node_complete` / `node_error` —
  `BaseNode.__call__()` emits those. Domain event names:
  - `pre_process_complete`
  - `input_validate_complete`
  - `retrieve_complete`
  - `rerank_filter_complete`
  - `generate_answer_complete`
  - `output_format_complete`
  - `post_process_complete`

  Three additional REJECT-path events exist (none is a second success-path
  event, so the "exactly ONE per success path" rule above still holds):
  - `pre_process_injection_blocked` — emitted by `PreProcessNode` when the
    prompt-injection screen refuses the string payload.
  - `input_validate_injection_blocked` — emitted by `InputValidateNode` when the
    same screen refuses `input_context.question`.
  - `post_process_degraded` — emitted by `PostProcessNode` when it renders the
    caller-facing sentence for a request declined upstream, in place of
    `post_process_complete`.

  The two injection payloads carry the matched PATTERN NAME (and the field
  name) only, and `post_process_degraded` carries the `error_code` marker only;
  the rejected text is never emitted.

## Insufficient-Evidence Decline

This is a life-safety-adjacent domain: `config/config.yaml` sets
`retrieval.score_threshold: 0.75` — materially stricter than a generic Cat 2
RAG default. `RerankFilterNode` drops every candidate below that floor before
`GenerateAnswerNode` ever sees it; when nothing survives, `GenerateAnswerNode`
returns an explicit insufficient-evidence decline (never falls back to
unsourced/parametric knowledge, never guesses a dose or diagnosis) that
recommends rephrasing the query or escalating to the clinical pharmacist /
quality-and-safety team.

The floor is a floor, not a caller preference: a request may RAISE
`score_threshold` for a single invocation, but a value below the configured one
is refused by `InputValidateNode` — the run ends with `error_code:
INVALID_REQUEST` and the correction sentence rather than being answered under
the weaker floor.

## Advisory Disclaimer

Every answer this template emits — including the insufficient-evidence
decline — carries the mandatory advisory line ("Advisory only —
physician/clinician judgment required..."). It is a hardcoded constant
appended by `OutputFormatNode` as part of the domain output contract, and it
is not driven by any prompt or config value a caller could suppress. The
output gate does not compose it — it independently VERIFIES that it is
present and blocks the answer if it is not, so the guarantee survives a future
change to the domain pipeline.

## v1 Implementation Note — LLM synthesis

v1 of this template is **deterministic end-to-end**: retrieval is keyword
scoring over the seeded KB and `GenerateAnswerNode` assembles the grounded
answer rule-based from the ranked passages (lead sentence + cited passage
excerpts). There is NO live model call and no model-client dependency in v1 —
`config/agent.yaml` therefore declares `generation_mode: "deterministic"` with
empty `requires.secrets` / `requires.extras`. The `llm` block in
`config/config.yaml` (`temperature: 0.0`, `max_tokens: 4000`) is forwarded
through `_parent_config()` for forward-compatibility but is not consumed by
any v1 node, and no `system_prompt` is read at runtime. The synthesis upgrade
seam is documented in `config/prompts/answer_synthesis_prompt.md`: a v2
`GenerateAnswerNode` swaps the rule-based assembly for a model call over the
same `ranked_documents` input and emits the same `grounded_answer` /
`citations` state contract, so no other node changes. The life-safety defaults
(deterministic generation, strict grounding, the insufficient-evidence
decline) must be preserved in any v2 implementation.

## Composition Pattern

- **Pattern:** `GraphNode` (subgraph) in the outer `main` slot.
- **Composition target:** `DomainWorkflowGraph` (inner `BaseGraph`).
- **Error propagation strategy:** `propagate` (inner errors re-raised as `SubgraphError`).
- Inner domain nodes run at `TrustLevel.ANONYMOUS`; outer pre/post_process run
  at `TrustLevel.VERIFIED_EXTERNAL`.

## Import Isolation Confirmation
- [x] Template does not import the platform-internal SDK.
- [x] Import targets: `framework/` and `shared/` only.
- [x] Base classes are framework base classes; no intermediate agent classes in
      any base position.

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| L1 base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | Fixed multi-step RAG workflow, no autonomous loop |
| Composition pattern | Standalone Cat 1 slots | GraphNode → inner BaseGraph | **GraphNode → inner BaseGraph** | 5-step domain workflow exceeds a single `main` node; nested keeps the outer backbone untouched |
| Node config contract | `execute(state, config=None)` | `execute(state)` + State-seeded config | **`execute(state)` + State-seeded config** | Nodes take no config parameter and no ctor args; `_extra_initial_state()` republishes the runtime `retrieval` values as scalar State fields instead |
| Answer synthesis | Rule-based assembly | Model call | **Rule-based** | Deterministic assembly is testable and auditable — required for a life-safety domain; a model call swaps in at the documented seam |
| KB storage | External vector store | Seeded JSON KB | **Seeded JSON KB** | Self-contained and deterministic; the retrieval contract (`retrieved_documents` JSON) is store-agnostic for a later vector-store upgrade |
| Relevance floor | Generic RAG default (`0.25`) | Healthcare life-safety floor | **`score_threshold: 0.75`** | Life-safety domains hold a stricter floor; this is the mechanism behind the mandatory insufficient-evidence decline |
| Advisory disclaimer placement | Composed by the output gate | Composed by OutputFormatNode, verified by the gate | **Composed by OutputFormatNode, verified by the gate** | The disclaimer is domain output, not a gate concern — but a guarantee that only one component upholds is one edit away from disappearing, so the gate re-checks it and fails closed |
| Caller `score_threshold` | Accept any value in range | Accept only values at or above the configured floor | **Tighten-only** | A caller who could lower the relevance floor could turn the mandatory decline off; refusing is the fail-closed reading |
| Ending a refused request | One terminal status for every refusal | Complete on a caller-correctable value, terminate on a refusal the agent makes | **Split by who can fix it** | Terminating on a mistyped `top_k` ends the conversation turn and leaves the reason only in the audit trail; completing with a reason marker and one plain sentence lets the caller correct and resend. A refusal the caller cannot act on — injection on the payload, an output-gate violation, a trust denial — must still terminate, or an unsafe run would be indistinguishable from an answer |

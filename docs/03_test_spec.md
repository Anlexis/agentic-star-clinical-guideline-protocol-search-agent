# Test Specification — HCR-C2-005

**Template ID:** HCR-C2-005
**Template Name:** ClinicalGuidelinesQAAgent
**Category:** Cat 2 (nested RAG) — life-safety domain (HCR)

This document is the test contract for the shipped implementation (state,
nodes, inner/outer graphs, configuration, server). The test code lives under
`tests/unit/` and `tests/proof_of_boundary/`.

## 1. Scope & Invocation Conventions

- Per-node unit tests for the 5 inner domain nodes + the 2 outer gate nodes.
- The caller-data contract on both input channels (`input_context` + the string
  payload).
- Configuration consistency (`config/agent.yaml` + `config/config.yaml` ↔ code)
  and seeded-KB integrity.
- Retrieval quality (golden queries over `config/kb/hcr_clinical_guideline_kb.json`),
  covering BOTH the grounded-answer path and the mandatory insufficient-evidence
  decline path.
- Inner-graph (`DomainWorkflowGraph`) and outer-graph
  (`ClinicalGuidelinesQAAgent`) composition / integration.
- Proof-of-Boundary: import isolation, State msgpack safety, invoke order
  (PB-6), HITL propagation (PB-7, conditional), server boot, and end-to-end
  behaviour through the real ASGI `POST /invoke`.

**Trust-gate invocation canon.** Every per-node test invokes the node via
`node(state)` — through `BaseNode.__call__`, which runs the trust gate → the
input mask / injection screen → `execute()` → the credential gate — never a
bare `node.execute(state)`. The state builder sets `caller_trust_level` to
`TrustLevel.VERIFIED_EXTERNAL.value` for the two outer gate slots
(PreProcessNode / PostProcessNode — the manifest's declared caller level) and
`TrustLevel.ANONYMOUS.value` for the five inner domain nodes.

**One deliberate exception — `TestNodeOwnsItsScreens`** (§2.1). Those cases
call `PreProcessNode().execute()` DIRECTLY, with no framework wrapper in front,
because they pin guarantees the TEMPLATE owns: the identifier strip and the
prompt-injection refusal must hold even where the framework's input gate is
absent or configured off. A refusal that only happens one layer up fails open
wherever that layer is missing.

**No config parameter.** Every `FunctionNode` implements EXACTLY
`execute(self, state) -> dict`. Retrieval tuning (`top_k` / `score_threshold` /
`kb_path`) is read from the SCALAR state fields `retrieval_top_k` /
`retrieval_score_threshold` / `retrieval_kb_path` (seeded by
`DomainWorkflowGraph._extra_initial_state()` in production); to exercise a knob,
tests seed the state key directly and still invoke through `node(state)`.

**Framework masking expectations.** The framework input gate masks
`user_input`/`validated_input`/`llm_response` (e-mail, phone/national-ID/card
digit groups, Title-Case name bigrams) to `[MASKED]` before `execute()` runs,
and refuses high-confidence prompt injection on the same fields. Positive-path
payloads are therefore lowercase clinical phrasing free of identifiers —
mandatory in this life-safety domain: no patient names, identifiers, or digit
groups anywhere in a test payload. Domain fields (`grounded_answer`,
`formatted_answer`, `retrieved_documents`, …) are not scan targets.
`PreProcessNode` then applies its OWN identifier strip and prompt-injection
screen inside `execute()` (see §2.1) — a positive-path payload must therefore
also avoid identifier LABELS ("patient id:", "生年月日", "〒", "患者名:") and
injection phrasing, or it will be redacted/rejected by design rather than by
accident.

Where BOTH layers can refuse the same payload, the assertions are
**behavioural** — the run ended in the refusal shape below and nothing was
carried forward — never the wording of whichever layer caught it first.

**Two endings for a refused request** (`docs/02_design.md` → "Rejection
Contract"), and the assertions distinguish them:

- **Rejected** (a value the caller can correct — empty input, any caller-data
  validation failure, injection on `input_context.question`): the run
  COMPLETES. Tests assert `status=SUCCESS`, a truthy `error_code`, that the
  product of the node is absent (no `validated_input` / no `search_query`),
  and that `error_log` names the offending field without echoing its value.
- **Terminated** (a refusal the agent makes on its own behalf — injection on
  the string payload, an output-gate violation, a trust denial): the run ends
  with `status=ERROR`. Tests assert `status=ERROR` and that nothing was carried
  forward.

Saying "rejected" in a row below means the first shape and "blocked" or
"denied" means the second; neither is ever asserted as the other.

**Audit muting.** `shared.*` is never sys.modules-stubbed (the framework
imports `shared.security` at load time). The domain audit emitter is muted via
an autouse fixture patching `src.nodes.<mod>.emit_trace_event`; the audit
assertion test re-patches the same attribute with a spy and asserts on
`call.args[1]` (the event payload).

**Life-safety relevance floor.** `RerankFilterNode`'s default
`score_threshold` is **0.75** (materially stricter than a generic Cat-2
default) — this is the mechanism behind the mandatory insufficient-evidence
decline. Retrieval-quality golden queries are phrased to clear that floor
deliberately (title-token-only phrasing), and `RerankFilterNode` fixtures use
scores either side of 0.75, never an arbitrary low bar.

## 2. Unit Test Cases

### 2.1 PreProcessNode (outer pre_process slot) — `test_pre_process_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| PRE-01 | Valid query | lowercase clinical question | `status=SUCCESS`, `validated_input` set, `enriched_context` carries channel/source |
| PRE-02 | Empty input | `""` / whitespace | **rejected**: `status=SUCCESS`, `error_log` non-empty, no `validated_input` |
| PRE-03 | Missing / non-string input | `user_input` absent; dict payload | **rejected**: `status=SUCCESS`; no `validated_input` for the dict payload |
| PRE-04 | Framework identifier screen | e-mail / 4-4-4 digit groups / JP phone → framework mask | raw identifier absent from `validated_input`; a redaction marker present |
| PRE-05 | Domain identifier strip | labelled patient id / MRN / カルテ番号, DOB (EN + JA), 〒postal, JP street address, labelled name (EN + JA), JP honorific name | raw identifier absent from `validated_input`; the class-labelled placeholder (`[REDACTED:PATIENT_ID]`, `[REDACTED:DOB]`, `[REDACTED:ADDRESS]`, `[REDACTED:NAME]`) present; surrounding clinical question intact |
| PRE-05b | Every class, via the helper | `_strip_direct_identifiers()` over EMAIL / PATIENT_ID / DOB / ADDRESS / MY_NUMBER / NATIONAL_ID / PHONE / NAME | each class reported in `hits`, raw value gone, placeholder present |
| PRE-06 | Prompt-injection screen | instruction override / system-prompt probe / role override / disclaimer bypass (EN + JA) | **terminated**: `status=ERROR`, no `validated_input`, `error_log` present and does NOT echo the payload |
| PRE-06b | Injection false positives | legitimate clinical phrasing ("when can we disregard the guideline recommendation", "which drugs act as a substrate", "指示された用量を無視してよい条件") | `status=SUCCESS` — never refused |
| PRE-07 | No PHI in the audit payload | multi-class PHI query | `pre_process_complete` payload carries `identifiers_redacted` + `identifier_classes` (names only); no redacted value appears in the serialised payload; `pre_process_injection_blocked` carries the pattern name only |
| PRE-08 | Domain audit | valid query | `pre_process_complete` emitted; payload (`call.args[1]`) carries `input_chars`, `identifiers_redacted=0`, `identifier_classes=[]`; all emits positional (3 args, no kwargs) |
| PRE-09 | Clinical text not mangled | doses ("500 mg", "1.25 mg/kg"), ICD-10 ("j18.9", "e11.9"), protocol ids ("protocol 7-2"), revision dates ("revised 2024-06-01"), ages ("72-year-old"), 患者さん/入院患者さん | payload returned byte-identical; no redaction marker |
| PRE-10 | **The node owns its screens** (`TestNodeOwnsItsScreens`) | every injection form, every identifier class, and the ordinary-question control set, each through `execute()` DIRECTLY (no framework wrapper) | injections **terminated** (`status=ERROR`, no `validated_input`, `error_log` names the pattern, no payload echo); identifiers redacted to their class placeholder with `status=SUCCESS`; ordinary clinical questions accepted; clinical numerals byte-identical |

> **Two screens run, in order.** (1) The framework `FunctionNode` input gate
> masks e-mail, JP/US phone, national-ID/card/My Number digit groups and Latin
> Title-Case name bigrams to `[MASKED]` BEFORE `execute()`, and refuses
> high-confidence prompt injection on the same fields. (2) `PreProcessNode`
> then runs its own screen inside `execute()`, redacting the direct identifiers
> the framework default does not model to class-labelled placeholders and
> refusing the injection forms the framework lets through. PRE-04 proves layer
> 1, PRE-05 proves layer 2 end-to-end through `node(state)`, PRE-05b proves the
> node's own patterns through the module-level helper, and **PRE-10 proves
> layer 2 with layer 1 removed entirely** — the template's guarantee must not
> be the framework's guarantee borrowed.
>
> **PRE-09 is a safety test, not a nicety.** A redaction rule that ate a dose,
> an ICD-10 code or a protocol number would be a worse clinical defect than the
> gap the screen closes, so every numeric rule is label-anchored or
> shape-specific and PRE-09 pins that contract.

### 2.2 InputValidateNode (inner node 1) — `test_input_validate_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| VAL-01 | Plain text | free-text query | whole string becomes `search_query`; filters `{category: None, top_k: None, score_threshold: None}` |
| VAL-02 | Whitespace | ragged spacing/newlines | collapsed to single spaces |
| VAL-03 | JSON envelope | `{"query","category","top_k"}` | all three parsed; `question` alias accepted; category lower-cased/stripped |
| VAL-04 | Malformed JSON | `{`-prefixed non-JSON | treated as plain-text query + parse note |
| VAL-05 | top_k out of range | 99 / −5 / 0 | **rejected** (`status=SUCCESS`, no `search_query`), `error_log` says `top_k must be between` |
| VAL-06 | top_k non-numeric | `"many"` | **rejected** (`status=SUCCESS`), `error_log` says `top_k must be a number` |
| VAL-07 | top_k fractional | `2.5` | **rejected** (`status=SUCCESS`), `error_log` says `whole number` |
| VAL-08 | Oversize query | > 2000 chars | truncated to 2000 + note |
| VAL-09 | Empty request | `""` | `search_query=""` + "empty request" note (non-fatal) |
| — | State shape | any | `query_filters` is a JSON string, never a bare dict |

> **VAL-05..07 fail CLOSED by design.** A silently clamped or silently dropped
> parameter answers a question the caller did not ask. Fail-closed is about the
> WORK, not the status: no `search_query` is produced, so nothing is retrieved
> and nothing is answered — while the run itself completes with the reason
> marker so the caller can correct the field and resend. The full contract —
> both channels, every field, the non-finite matrix — is §2.10.

### 2.3 RetrieveNode (inner node 2) — `test_retrieve_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RET-01 | Happy path | CAP/antibiotic-stewardship query (= deploy payload) | top-1 candidate is `kb-001` |
| RET-02 | Ordering | same query | scores strictly sorted desc; all > 0 |
| RET-03 | Entry shape | any hit | keys `{id,title,category,source,score,excerpt}`; excerpt ≤ 400 chars |
| RET-04 | Category filter | `query_filters.category="renal_dosing"`, query `"renal"` | both `renal_dosing` entries returned (title + content hit); top-1 `kb-002` |
| RET-05 | Empty query | `""` | no candidates |
| RET-06 | State-seeded `kb_path` override | `retrieval_kb_path` set to a bogus path, via `node(state)` | `[]` + "not readable" note |
| RET-07 | Unseeded fallback | no `retrieval_kb_path` in state | resolves the real seeded KB via the module default |
| RET-08 | Notes accumulation | prior `intake_notes` | appended, never clobbered |

### 2.4 RerankFilterNode (inner node 3) — `test_rerank_filter_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RRF-01 | HCR relevance floor | scores 0.9 / 0.5 | 0.5 dropped (default **0.75** floor — a real, non-trivial score is still declined) |
| RRF-02 | State-seeded `score_threshold` override | `retrieval_score_threshold=0.5`, via `node(state)` | 0.6 now survives the lowered floor |
| RRF-03 | State-seeded `top_k` override | `retrieval_top_k=1`, via `node(state)` | one survivor, highest score |
| RRF-04 | Category boost | matching category | +0.1, re-ranked ahead (both pre-clear 0.75 so the boost drives reordering, not a threshold flip) |
| RRF-05 | Boost cap | 0.95 + boost | capped at 1.0 |
| RRF-06 | Caller top_k | stricter (1) wins; looser (10) does not widen | enforced |
| RRF-07 | Garbage entries | non-dict / uncoercible score | skipped / coerced to 0.0 and dropped |
| RRF-08 | Tie-break | equal scores | deterministic id-ascending order |

### 2.5 GenerateAnswerNode (inner node 4) — `test_generate_answer_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| GEN-01 | Citation markers | 2 ranked passages | `[1]`/`[2]` markers with titles |
| GEN-02 | Lead sentence | query present | query quoted in the lead |
| GEN-03 | Citations list | ranked passages | refs 1..n mirror ranked order; id/title/source carried |
| GEN-04 | Groundedness | single passage | answer body traces to ranked passages only |
| GEN-05 | **Insufficient-evidence decline** | empty/missing `ranked_documents` | explicit escalation answer ("does not contain sufficient coverage" + escalate to clinical pharmacist); `citations=[]`; never falls back to unsourced/parametric knowledge |

### 2.6 OutputFormatNode (inner node 5, terminal) — `test_output_format_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| FMT-01 | Full compose | body + citations | header + body + `## Sources` rows + advisory disclaimer; `status=SUCCESS` |
| FMT-02 | Blank source | citation without source | no `()` suffix |
| FMT-03 | Disclaimer | every input, incl. the decline | disclaimer rides with every answer — non-suppressible |
| FMT-04 | No citations | empty list | explicit "- none (…)" sources line |
| FMT-05 | Missing body | no `grounded_answer` | fallback text; `status=SUCCESS` |

### 2.7 PostProcessNode (outer post_process slot; the output gate) — `test_post_process_node.py`

The gate enforces three invariants, and every one is tested BOTH ways — a
violating output is blocked, a compliant output passes byte-identical.

| ID | Case | Input (`result`) | Expected |
|----|------|------------------|----------|
| POST-01 | Clean output | full rendered answer (body + Sources + disclaimer) | `formatted_output=result`, `status=SUCCESS` |
| POST-02 | Empty result | `""` | forwarded as-is, `status=SUCCESS` (non-fatal) |
| POST-03..06 | Credential leak | `sk-` API key / `password=` assignment / JWT (built at runtime) / Bearer token | `formatted_output` + `result` replaced with the sanitised stub, `status=ERROR`, raw secret absent from both |
| POST-07 | Identifier leak | `MRN 4451233` / `MRN-4451233` / `123-45-6789` | blocked; the digits are absent from the surfaced fields entirely — redacted, never rewritten into a shape the scan would miss |
| — | Nested structures | violation nested in a dict/list result | caught exactly like a top-level string (the scan recurses) |
| POST-08 | Disclaimer missing | rendered answer with the advisory line removed | blocked, `status=ERROR`, error names the advisory disclaimer |
| POST-09 | Unlisted citation | body cites `[2]`, Sources lists only `[1]` | blocked, `status=ERROR`, error names the Sources section |
| — | Grounded / declined answers | markers all listed, or no markers at all | `status=SUCCESS` |

### 2.8 Configuration consistency — `test_config_manifest.py`

| ID | Case | Expected |
|----|------|----------|
| CFG-01 | Flat manifest | `id=HCR-C2-005`, `enabled=true`, and NO `agent:` block (registry keys sit at root level) |
| CFG-02 | Class-name contract | manifest `class` == `src.graph.graph.ClinicalGuidelinesQAAgent`; `name` == the graph's `name` |
| CFG-03 | Classification | Cat 2 / HCR / namespace `hcr` / RAGAgent / `generation_mode: deterministic` |
| CFG-04 | Trust level | manifest `VERIFIED_EXTERNAL` == PreProcessNode & PostProcessNode `required_trust_level` |
| CFG-05 | Compile-time requirements | `requires.secrets` and `requires.extras` are both `[]` — the pipeline constructs no model client and requires no secret |
| CFG-06 | Runtime parameters | `max_retry` int in `0 ≤ v < 10` (framework ceiling); `timeout_s` positive int; hitl not enabled (PB-7 waiver contract) |
| CFG-07 | Retrieval block | `top_k=8`/`score_threshold=0.75` mirror `RetrieveNode`/`RerankFilterNode` module defaults; `kb_path` exists |
| CFG-08 | `_parent_config()` | forwards the `config/config.yaml` retrieval + llm blocks, never `{}`; `timeout_s` renamed to `timeout_seconds` |
| — | `load_runtime_config()` | reads the live `config/config.yaml` |
| — | KB integrity | JSON list ≥ 5 entries; unique ids; required keys per entry |

### 2.10 Caller-data contract — `test_caller_data_contract.py`

Both input channels, field by field. Inputs are hostile until proven bounded.

Every "rejected" / "refused" row below is asserted through one shared helper,
which pins the completing shape exactly: `status=SUCCESS`, a truthy
`error_code`, no `search_query` in the returned dict, and the offending field
named in `error_log`. Nothing is processed; the run ends carrying the reason.

| ID | Case | Expected |
|----|------|----------|
| CDC-01 | `input_context.question` | becomes `search_query`; wins over the string payload; absent caller data degrades to the payload |
| CDC-02 | Structured filters | `category` / `top_k` / `score_threshold` carried into `query_filters` |
| CDC-03 | **Non-finite matrix, per numeric field** | `"NaN"` / `"Infinity"` / `"-Infinity"` / raw `float("nan")` / raw `float("inf")` / raw `float("-inf")` / `True` / free text / list / dict — every one **rejected**, on `top_k` AND `score_threshold`, on the `input_context` AND the JSON-envelope channel |
| CDC-04 | Out-of-range | `top_k` ∉ [1,20], `score_threshold` ∉ [0.0,1.0] → rejected |
| CDC-05 | Relevance floor | a value below the configured floor is **refused**; a stricter value is accepted; a corrupt seeded floor falls back to the module default rather than opening the gate |
| CDC-06 | Inert category grammar | free text / punctuation / >32 chars / non-string → rejected; valid identifiers normalised to lowercase |
| CDC-07 | Context-channel screens | identifiers in `input_context.question` redacted to their class placeholder; prompt injection on that channel refused without echoing the payload |
| CDC-08 | No value echo | the rejected value never appears anywhere in the returned dict |
| — | Render-safe normalisation | Markdown metacharacters and line breaks removed from the echoed question; clinical punctuation (hyphens, parentheses) preserved |

### 2.9 Retrieval quality (golden queries) — `test_retrieval_quality.py`

| ID | Case | Expected |
|----|------|----------|
| QUAL-01 | 6 golden domain queries | expected KB entry is top-1 (kb-001/002/004/006/007/010) |
| QUAL-02 | Relevance floor | every survivor ≥ 0.75 |
| QUAL-03 | Citation integrity | every survivor id exists in the seeded KB |
| QUAL-04 | Precision | CAP query keeps ONLY `kb-001` |
| QUAL-05 | Category filter + floor interaction | `renal_dosing` filter returns only `kb-002` — `kb-009` shares the category but its content-only match does not clear 0.75 |
| QUAL-06 | No coverage | out-of-domain query → zero survivors |
| QUAL-07 | Escalation answer | no-coverage → explicit escalation text, no citations |

## 3. Integration / Composition

### 3.1 Inner graph — `test_domain_workflow_graph.py`

| ID | Case | Expected |
|----|------|----------|
| INT-01 | Composition | inherits `BaseGraph`; registers exactly the 5 domain nodes; no initialize/finalize |
| INT-02 | Config + context forwarding | `_extra_initial_state()` republishes the retrieval block as THREE scalar fields (`retrieval_top_k`/`retrieval_score_threshold`/`retrieval_kb_path`) — never a JSON blob — and seeds the bridged `input_context`; `_validate_config()` rejects a non-positive / non-integer `timeout_s` |
| INT-03 | Output shape | `get_output()` emits `formatted_answer`/`citations`/`status`/… (the merge contract); `route()` → END on error |
| INT-04 | Inner e2e | full inner `invoke()` → SUCCESS; formatted answer + disclaimer + kb-001 citation for the CAP query; no-coverage query still terminates SUCCESS with the escalation text; inner `node_history` = the 5 domain nodes in linear order |

### 3.2 Outer graph + e2e — `test_graph_composition.py`

| ID | Case | Expected |
|----|------|----------|
| INT-05 | Outer composition | inherits `AgentBaseGraph` directly; `Graph` alias; `add_edges()` NOT overridden |
| INT-06 | Backbone slots | compile() fills all 5; pre/main/post are PreProcessNode / ClinicalGuidelineSearchGraphNode / PostProcessNode |
| INT-07 | `get_subgraph()` | returns `DomainWorkflowGraph` carrying the forwarded retrieval config |
| INT-08 | `extract_input()` | prefers `validated_input`, falls back to `user_input` |
| INT-09 | `merge_output()` | inner `formatted_answer` → outer `guideline_answer` AND `result`; `citations`/`status` mapped; changed keys only |
| INT-10 | Runtime-config fallback | `_parent_config()` never `{}` even with an unreadable `config/config.yaml`; fallback `score_threshold=0.75` |
| INT-11 | e2e happy path | VERIFIED_EXTERNAL invoke → SUCCESS; `output` = gated formatted answer; PostProcessNode traversed |
| INT-12 | e2e trust denial | ANONYMOUS invoke → ERROR; empty `output`; PostProcessNode NOT traversed |
| INT-13 | **Runtime values are live** | a tightened `score_threshold` in `config/config.yaml` changes the answer end-to-end (nothing clears the floor → the decline); a non-positive `timeout_s` fails at `compile()` |
| — | **e2e insufficient-evidence decline** | out-of-domain query, VERIFIED_EXTERNAL invoke → **SUCCESS** (declining is a correct outcome, not an error) with the explicit escalation text in `output` |
| — | State helpers | `to_json`/`from_json` round-trip; None/malformed handling |

## 4. Proof-of-Boundary

| ID | Case | Expected |
|----|------|----------|
| PB-IMPORT | `test_import_isolation.py` | no platform-internal `agenticstar` import anywhere under `src/` |
| PB-STATE | `test_state_safety.py` | `State` has no credential-named fields and no `BaseModel` / `InvocationContext` annotations |
| PB-6 | `test_pb_invoke_order.py` | full `Graph().invoke()` with `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` (never `for_internal()`) over the payload byte-equal to `deploy/invoke_payload.json`'s `input` → SUCCESS with outer `node_history` exactly `[InitializeNode, PreProcessNode, ClinicalGuidelineSearchGraphNode, PostProcessNode, FinalizeNode]` |
| PB-7 | `test_pb7_hitl_interrupt_propagation.py` | **Auto-waived — non-HITL** (no graph class declares `propagate_hitl=True`; `config/config.yaml` has no `hitl.enabled: true`); dynamic-detection skip stub |
| PB-BOOT | `test_server_boot.py` | `import src.api.server` does not raise; module-level agent is this template's class, compiled; fresh ctor→`compile()` fills the 5 backbone slots; `/health` reports the agent |
| PB-E2E | `test_invoke_e2e.py` | the real ASGI `POST /invoke` with Bearer auth: bridged `input_context` produces a cited, non-empty answer that a different question changes; the category filter and a tightened floor change the outcome; an uncovered question declines; every malformed field, and injection on the context channel, are **rejected** — HTTP 200, `status=success`, and `output` carrying the caller-facing correction sentence rather than an answer, with the rejected value never echoed anywhere in the response; oversized `input_context` → 413; a missing token → 401; and an output scan for the disclaimer, citation grounding, identifier survival, and structure forgery |

> **Mandatory set:** PB-IMPORT, PB-STATE, PB-6, PB-BOOT and PB-E2E. PB-7
> applies only to HITL-enabled templates — this template is non-HITL, so PB-7
> is **Auto-waived — non-HITL** and its skip is expected.

## 5. Test Execution Summary

- Runner: the real `agenticstar-agentcore==1.0.1` wheel — a stub pass is not a
  pass — mirroring the two steps the test job runs:
  - `python -m pytest tests/` (full tree, includes `proof_of_boundary/`):
    **300 passed, 1 skipped, 0 failed** (301 collected)
  - `python -m pytest tests/proof_of_boundary/` (boundary subset re-run):
    **37 passed, 1 skipped, 0 failed** (38 collected; already included in the
    total above)
- Total unique tests: 301 — Pass: 300 / Fail: 0 / Skip: 1 (PB-7 conditional
  stub — auto-waived, non-HITL; the single skip appears in both step counts
  because PB-7 lives under `tests/proof_of_boundary/`, which the second step
  re-runs)
- Non-vacuity: the caller-context bridge was disabled and the boundary suite
  re-run — **15 of the end-to-end cases failed**, then passed again with the
  bridge restored. They detect the behaviour they claim to, rather than
  restating whatever the code happens to do.
- Type and style: `mypy src` → Success (19 files); `ruff check` and
  `ruff format --check` clean on `src/` and `tests/` at the pinned version.
- Determinism: no model call, no network; retrieval + answer assembly are
  rule-based.
- Coverage: both mandatory life-safety paths are exercised end-to-end — the
  grounded-answer path (community-acquired-pneumonia query, `kb-001`) and the
  insufficient-evidence decline path (out-of-domain query, zero survivors of
  the 0.75 relevance floor).

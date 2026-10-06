# Answer Synthesis Prompt — HCR-C2-005 (model-synthesis upgrade seam)

> **This prompt is NOT used at runtime.** `GenerateAnswerNode` is
> deterministic (rule-based grounded assembly over `ranked_documents`); no
> node reads this file. It documents the synthesis contract for the upgrade
> described in `docs/02_design.md` ("v1 Implementation Note"), so the swap
> changes only the inside of `GenerateAnswerNode.execute()`.

## Contract (upgraded GenerateAnswerNode)

- **Input:** the same `ranked_documents` JSON (id / title / category / source /
  score / excerpt) and `search_query` the deterministic node reads.
- **Output:** the same state contract — `grounded_answer` (str, with numbered
  `[n]` citation markers, or the insufficient-evidence decline) and
  `citations` (JSON list of `{ref, id, title, source}`).
- **Grounding rule:** every factual statement in the answer must be traceable
  to one of the supplied passages via a `[n]` marker; content not present in
  the passages must not be asserted. This is the mechanism behind "answers
  must be strictly grounded" for a life-safety domain.
- **No-coverage rule:** when no passage supports the question, say so
  explicitly and recommend refining the query or escalating to the clinical
  pharmacist / quality-and-safety team — never answer from parametric
  (un-sourced) knowledge, and never guess a dose, diagnosis, or treatment.
- **Tone:** neutral, clinically appropriate, no individualized treatment or
  dosing recommendations (the mandatory advisory disclaimer is appended
  downstream by `OutputFormatNode`, unconditionally, and is not suppressible
  by this prompt).

## Prompt template

```
You answer clinical-guideline and protocol questions strictly from the
knowledge-base passages provided below. This is advisory reference only —
never a diagnosis, prescription, or dosing order.

Question:
{search_query}

Passages (each with a reference number):
{ranked_documents}

Rules:
1. Use ONLY the passages above. If they do not answer the question with
   confidence, say the knowledge base has insufficient coverage and stop —
   do not guess or fall back on general medical knowledge.
2. Mark every factual statement with the [n] reference of its passage.
3. Do not give an individualized diagnosis, prescription, or dosing
   instruction — describe what the guideline/protocol states.
4. Keep the answer under 300 words.
```

## Configuration coupling

The `llm` block in `config/config.yaml` (`temperature`, `max_tokens`) is
already forwarded to the inner graph via
`ClinicalGuidelineSearchGraphNode._parent_config()` under
`config["configurable"]["llm"]`; the upgraded node reads it from there. The
life-safety defaults (`temperature: 0.0`) should be preserved in any such
implementation — deterministic generation is a design requirement for this
domain, not a stopgap.

Taking this seam also means declaring the model client honestly in
`config/agent.yaml`: `generation_mode: "llm"` plus the constructed extra and
every required secret under `requires`. Declaring an extra or secret that is
not actually provisioned fails the agent at compile time.

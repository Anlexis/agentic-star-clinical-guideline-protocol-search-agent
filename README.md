# Clinical Guideline & Protocol Search Agent

AI agent for searching clinical guidelines and protocols, built with Agentic Star.

> **Category**: Cat 2 (domain-specific pipeline)
> **Industry**: Healthcare
> **Template ID**: HCR-C2-005

## Overview

Answers clinical questions from a curated guideline and protocol knowledge base, with
citations. A clinician, pharmacist or quality-and-safety reviewer asks a natural-language
question — empiric therapy, dose adjustment, contraindication checks, protocol steps — and the
agent retrieves the most relevant guideline passages, assembles a grounded answer with numbered
citations, and lists its sources. It answers only from the knowledge base: when no passage clears
the relevance floor it says so explicitly and recommends escalation rather than answering from
unsourced knowledge, and every answer — including that decline — carries a non-suppressible
advisory notice that clinician judgment is required. The pipeline is deterministic (keyword
retrieval and rule-based answer assembly over a seeded knowledge base) — no live model call, no
vector store, no patient-record integration.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails at graph
compile / start-up preflight rather than starting in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent manifest, runtime parameters, knowledge base, prompts
docs/         design specification and test specification
```

See `docs/` for the design specification and the test specification.

## Configuration

| File | Purpose |
|---|---|
| `config/agent.yaml` | Static manifest read by the platform registry — identity, category, entry-point class, required trust level, compile-time requirements. Every key sits at root level. |
| `config/config.yaml` | Runtime parameters passed to the graph constructor — `max_retry`, `timeout_s`, the `retrieval` tuning block (`top_k`, `score_threshold`, `kb_path`) and the reserved `llm` block. |
| `config/kb/` | The seeded guideline knowledge base the agent answers from. |

`retrieval.score_threshold` is the relevance floor: passages below it are dropped and the agent
declines rather than answering from weak evidence. A caller may raise it for a single request, but
never lower it.

## Calling the agent

`POST /invoke` accepts the question as plain text and, optionally, structured parameters in
`input_context`:

```json
{
  "input": "search the clinical-guideline knowledge base for the attached question.",
  "input_context": {
    "question": "what is the first-line empiric antibiotic for community-acquired pneumonia?",
    "category": "infectious_disease",
    "top_k": 3,
    "score_threshold": 0.85
  }
}
```

Every field is validated before use: numbers must be finite and in range, `category` must match
`[a-z0-9_]{1,32}`, and a rejected request names the offending field without echoing its value.

## Customising

1. Adjust `config/config.yaml` for your own retrieval tuning and policies.
2. Replace `config/kb/` with your own guideline corpus (same entry shape).
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.

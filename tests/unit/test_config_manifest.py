# HCR-C2-005 — Unit Tests: manifest / config consistency
#
# Two files, two jobs:
#   config/agent.yaml  — the static manifest the platform registry reads. Every
#                        key sits at ROOT level and `class:` is a single dotted
#                        import path; the declared class must BE the
#                        src/graph/graph.py agent class.
#   config/config.yaml — the runtime parameters. Live configuration, not
#                        documentation: ClinicalGuidelineSearchGraphNode
#                        ._parent_config() forwards its retrieval/llm blocks
#                        into the inner graph, and src/api/server.py passes the
#                        whole file into the agent constructor.
#
# These tests pin config ↔ code consistency so a drift fails fast.
#
# Mirrors docs/03_test_spec.md §2.8 (CFG-01..CFG-08).
# Deterministic — no LLM, no network.

import json
import pathlib

import yaml

from framework.schemas.trust_level import TrustLevel

from src.graph.graph import (
    ClinicalGuidelineSearchGraphNode,
    ClinicalGuidelinesQAAgent,
    load_runtime_config,
)
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = yaml.safe_load((_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
_RUNTIME = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


class TestManifestIdentity:
    def test_cfg_01_manifest_is_flat_and_identifies_the_template(self):
        assert _MANIFEST["id"] == "HCR-C2-005"
        assert _MANIFEST["enabled"] is True
        assert "agent" not in _MANIFEST, "registry keys must sit at root level, not under agent:"

    def test_cfg_02_declared_class_is_the_graph_class(self):
        # Class-name contract: manifest class == graph.py class == server import.
        assert _MANIFEST["class"] == "src.graph.graph.ClinicalGuidelinesQAAgent"
        assert _MANIFEST["class"].rsplit(".", 1)[-1] == ClinicalGuidelinesQAAgent.__name__
        assert _MANIFEST["name"] == ClinicalGuidelinesQAAgent().name

    def test_cfg_03_classification(self):
        assert _MANIFEST["category"] == "Cat 2"
        assert _MANIFEST["industry"] == "HCR"
        assert _MANIFEST["namespace"] == "hcr"
        assert _MANIFEST["base_type"] == "RAGAgent"
        assert _MANIFEST["generation_mode"] == "deterministic"


class TestManifestSecurity:
    def test_cfg_04_required_trust_level_matches_outer_gate_nodes(self):
        declared = TrustLevel(_MANIFEST["required_trust_level"])
        assert declared is TrustLevel.VERIFIED_EXTERNAL
        assert PreProcessNode.required_trust_level is declared
        assert PostProcessNode.required_trust_level is declared

    def test_cfg_05_compile_time_requirements_are_declared_and_empty(self):
        # The pipeline is deterministic: it constructs no model client and
        # requires no secret. Declaring one that is not provisioned fails the
        # agent at compile time in a real deployment.
        assert _MANIFEST["requires"]["secrets"] == []
        assert _MANIFEST["requires"]["extras"] == []

    def test_hitl_is_not_enabled(self):
        # PB-7 auto-waiver contract: this template declares no HITL.
        assert (_RUNTIME.get("hitl") or {}).get("enabled", False) is False


class TestRuntimeConfig:
    def test_cfg_06_runtime_parameters_are_within_framework_bounds(self):
        max_retry = _RUNTIME["max_retry"]
        assert isinstance(max_retry, int) and not isinstance(max_retry, bool)
        assert 0 <= max_retry < 10  # framework MAX_RETRY_CEILING
        timeout_s = _RUNTIME["timeout_s"]
        assert isinstance(timeout_s, int) and not isinstance(timeout_s, bool) and timeout_s > 0

    def test_cfg_07_retrieval_block_matches_node_defaults(self):
        # Node module defaults mirror config/config.yaml — a drift silently
        # changes tuning. Life-safety relevance floor: score_threshold 0.75.
        retrieval = _RUNTIME["retrieval"]
        from src.nodes.rerank_filter_node import _DEFAULT_SCORE_THRESHOLD
        from src.nodes.rerank_filter_node import _DEFAULT_TOP_K as rerank_top_k
        from src.nodes.retrieve_node import _DEFAULT_KB_PATH
        from src.nodes.retrieve_node import _DEFAULT_TOP_K as retrieve_top_k

        assert retrieval["top_k"] == retrieve_top_k == rerank_top_k == 8
        assert retrieval["score_threshold"] == _DEFAULT_SCORE_THRESHOLD == 0.75
        assert retrieval["kb_path"] == _DEFAULT_KB_PATH
        assert (_ROOT / retrieval["kb_path"]).is_file()

    def test_cfg_08_parent_config_forwards_the_runtime_blocks(self):
        cfg = ClinicalGuidelineSearchGraphNode()._parent_config()
        configurable = cfg["configurable"]
        assert configurable["retrieval"] == _RUNTIME["retrieval"]
        assert configurable["llm"] == _RUNTIME["llm"]
        assert configurable["retrieval"], "_parent_config() must never forward an empty retrieval block"
        # timeout_s is renamed to the key the inner graph validates.
        assert configurable["max_retry"] == _RUNTIME["max_retry"]
        assert configurable["timeout_seconds"] == _RUNTIME["timeout_s"]

    def test_load_runtime_config_reads_the_live_file(self):
        assert load_runtime_config() == _RUNTIME


class TestSeededKnowledgeBase:
    def test_kb_is_a_well_formed_entry_list(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        assert isinstance(entries, list)
        assert len(entries) >= 5, "seeded KB must carry a usable corpus"
        for entry in entries:
            assert set(entry.keys()) == {"id", "title", "category", "source", "tags", "content"}
            assert entry["id"] and entry["title"] and entry["content"]

    def test_kb_ids_are_unique(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        ids = [e["id"] for e in entries]
        assert len(ids) == len(set(ids))

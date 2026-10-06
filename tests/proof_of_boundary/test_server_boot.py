# PB-BOOT - Server Boot Boundary: importing src.api.server must not raise.
#
# The standalone entry point (src/api/server.py) constructs the agent, calls
# compile(), and provisions secrets AT MODULE LEVEL - so an import failure is
# a deploy-time outage that unit tests on nodes would never catch (this exact
# class of failure surfaces only once the app is actually started). The
# boundary proven here:
#
#   1. `import src.api.server` completes (FastAPI app + compiled agent built,
#      DotenvProvider tolerates absent env/ tier files).
#   2. The module-level agent is THIS template's agent class, compiled.
#   3. A fresh agent constructs + compiles via the supported path
#      (compile() -> register_nodes() -> _build_graph()).
#
# docs/03_test_spec.md section 4 (PoB).
# Deterministic - no LLM, no network, no socket bind (the ASGI app is built
# but never served).

import importlib


class TestServerBoot:
    """PB-BOOT: the standalone HTTP entry point must import and compile."""

    def test_server_module_imports_without_raising(self):
        server = importlib.import_module("src.api.server")
        assert server.app is not None, "FastAPI app must be constructed at import"

    def test_module_level_agent_is_compiled(self):
        server = importlib.import_module("src.api.server")
        from src.graph.graph import ClinicalGuidelinesQAAgent

        assert isinstance(server.agent, ClinicalGuidelinesQAAgent)
        assert server.agent._compiled is not None, "server.py must compile() the agent at import time"

    def test_health_endpoint_reports_this_agent(self):
        server = importlib.import_module("src.api.server")
        payload = server.health()
        assert payload == {"status": "ok", "agent": "ClinicalGuidelinesQAAgent"}

    def test_fresh_agent_constructs_and_compiles(self):
        """The supported construction path: ctor (no args) -> compile()."""
        from src.graph.graph import ClinicalGuidelinesQAAgent

        agent = ClinicalGuidelinesQAAgent()
        agent.compile()
        assert agent._compiled is not None
        assert set(agent._nodes.keys()) == {
            "initialize",
            "pre_process",
            "main",
            "post_process",
            "finalize",
        }

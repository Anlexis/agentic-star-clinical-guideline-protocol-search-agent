# HCR-C2-005 — unit-test fixtures
#
# Mutes the domain audit emitter in every src.nodes module. The framework
# imports shared.security at load time, so sys.modules-stubbing shared.* would
# break collection — instead each node module's already-imported
# emit_trace_event reference is monkeypatched to a no-op. A test that asserts
# on the audit payload re-patches the same module attribute with its own spy
# (the later patch wins for that test).
#
# The framework's OWN lifecycle events (emitted by framework.nodes.base_node)
# are left untouched — they are fire-and-forget
# log lines and part of the behaviour under test.

import pytest

import src.nodes.generate_answer_node
import src.nodes.input_validate_node
import src.nodes.output_format_node
import src.nodes.post_process_node
import src.nodes.pre_process_node
import src.nodes.rerank_filter_node
import src.nodes.retrieve_node

_AUDITED_NODE_MODULES = (
    src.nodes.generate_answer_node,
    src.nodes.input_validate_node,
    src.nodes.output_format_node,
    src.nodes.post_process_node,
    src.nodes.pre_process_node,
    src.nodes.rerank_filter_node,
    src.nodes.retrieve_node,
)


@pytest.fixture(autouse=True)
def mute_domain_audit(monkeypatch):
    """Silence the domain audit emitter in every node module."""
    for module in _AUDITED_NODE_MODULES:
        monkeypatch.setattr(module, "emit_trace_event", lambda *args, **kwargs: None)

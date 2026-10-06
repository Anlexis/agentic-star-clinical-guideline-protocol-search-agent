"""Framework compliance: the FunctionNode security gates must not be bypassable.

Required for every template that has one or more FunctionNode subclasses.
They validate the runtime enforcement boundary: domain nodes may extend the
default input/output gates only through _extra_security_gate_input() and
_extra_security_gate_output(), never by replacing them.
"""

import pytest

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel


class TestFunctionNodeFinalSecurityGates:
    """TC-06/TC-07: FunctionNode security gates are non-bypassable."""

    def test_tc06_security_gate_input_cannot_be_overridden(self):
        """Overriding the default input gate raises at class definition."""
        with pytest.raises(TypeError, match="_security_gate_input"):

            class _InvalidInputGateOverride(FunctionNode):
                required_trust_level = TrustLevel.ANONYMOUS

                def _security_gate_input(self, state):
                    return state

                def execute(self, state):
                    return {"status": AgentStatus.SUCCESS.value}

    def test_tc07_security_gate_output_cannot_be_overridden(self):
        """Overriding the default output gate raises at class definition."""
        with pytest.raises(TypeError, match="_security_gate_output"):

            class _InvalidOutputGateOverride(FunctionNode):
                required_trust_level = TrustLevel.ANONYMOUS

                def _security_gate_output(self, result):
                    return result

                def execute(self, state):
                    return {"status": AgentStatus.SUCCESS.value}

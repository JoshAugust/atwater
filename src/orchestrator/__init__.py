"""
src.orchestrator — Context assembler, flow controller, and tool loader.

The orchestrator is the cognitive architecture's coordination layer.
It decides WHAT each agent sees (context_assembler), WHEN agents run
(flow_controller), and WHICH tools are available per turn (tool_loader).

Public surface
--------------
Types::

    from src.orchestrator import AgentContext, AgentResult, CycleResult

Context assembly::

    from src.orchestrator import ContextAssembler

Flow control::

    from src.orchestrator import FlowController

Tool loading::

    from src.orchestrator import ToolLoader

Typical workflow::

    from src.orchestrator import ContextAssembler, FlowController, ToolLoader
    from src.memory import SharedState, KnowledgeBase
    from src.optimization import create_study, DEFAULT_SEARCH_SPACE

    study = create_study()
    state = SharedState()
    kb = KnowledgeBase()

    assembler = ContextAssembler(shared_state=state, knowledge_base=kb, study=study)
    controller = FlowController(
        shared_state=state,
        knowledge_base=kb,
        study=study,
        search_space=DEFAULT_SEARCH_SPACE,
        context_assembler=assembler,
    )

    result = controller.run_cycle(cycle_number=1)
    print(result.score)
"""

from .context_assembler import AgentContext, AgentResult, ContextAssembler
from .flow_controller import CycleResult, FlowController
from .tool_loader import ToolLoader

__all__ = [
    # Types
    "AgentContext",
    "AgentResult",
    "CycleResult",
    # Core classes
    "ContextAssembler",
    "FlowController",
    "ToolLoader",
]

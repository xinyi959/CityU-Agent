from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from agent.state.citation import Citation


class ProgrammeRefState(TypedDict, total=False):
    programme_id: str
    programme_name: str


class InputState(TypedDict, total=False):
    """Public input contract for ``app.invoke`` / ``app.stream``.

    Two entry paths are supported:

    - chat UI passes ``messages`` (the full conversation; ``input_adapter``
      extracts the latest user turn into ``query``);
    - CLI / direct callers pass ``query`` directly.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    query: str


class OutputState(TypedDict, total=False):
    """Public output contract for ``app.invoke``.

    Only the user-facing result is exposed. Internal routing / retrieval /
    generation intermediates (``intent``, ``decisions``, ``evidence``,
    ``answer``, ``programme_ref``, ...) are deliberately NOT part of the
    public result; they are observable via ``stream_mode="values"`` or
    ``get_state()`` when debugging.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    final_response: str
    citations: list[Citation]


class AgentState(InputState, OutputState, total=False):
    """Internal full state shared by every node.

    The public input/output boundaries are narrowed by :class:`InputState`
    and :class:`OutputState`; this schema additionally carries the routing,
    retrieval and generation intermediates that flow between nodes.
    """

    # Router output. For compound queries the full plan lives in
    # ``decisions`` and is consumed by the dispatcher.
    intent: str

    # Router sub-decisions, one per sub-question (list of RouterSubDecision
    # dicts). Absent on pre-Phase-1 checkpoints.
    decisions: list

    # Retrieval evidence
    evidence: list

    # Generated answer
    answer: str

    # Router extracted reference
    programme_ref: ProgrammeRefState

    # Confirmed programme reference. Retrievers write this back after
    # find_programme succeeds (with programme_id filled); later turns reuse it
    # so an omitted referent does not force a repeat of name -> id inference.
    resolved_programme_ref: ProgrammeRefState

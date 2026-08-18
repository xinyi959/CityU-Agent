from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from agent.models import Citation

class ProgrammeRefState(TypedDict, total=False):
    programme_id: str
    programme_name: str

class AgentState(TypedDict, total=False):

    # Chat UI / LangGraph messages
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

    # User query
    query: str

    # Router output. For compound queries, retrieval_type/field reflect the
    # FIRST sub-decision (back-compat with the single-decision graph); the
    # full plan lives in `decisions` and is consumed by the dispatcher.
    intent: str
    retrieval_type: str
    field: str | None

    # Router sub-decisions, one per sub-question (list of RouterSubDecision
    # dicts). Absent on pre-Phase-1 checkpoints.
    decisions: list

    # Retrieval evidence
    evidence: list

    # Generated answer
    answer: str

    # Citation output
    citations: list[Citation]

    # Final UI response
    final_response: str
    
    # Router extracted reference
    programme_ref: ProgrammeRefState

    # Confirmed programme reference. Retrievers write this back after
    # find_programme succeeds (with programme_id filled); later turns reuse it
    # so an omitted referent does not force a repeat of name -> id inference.
    resolved_programme_ref: ProgrammeRefState

    
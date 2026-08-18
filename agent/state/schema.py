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

    # Router output
    intent: str
    retrieval_type: str
    field: str | None

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

    
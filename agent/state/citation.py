from typing import TypedDict


class Citation(TypedDict):
    """Structured citation item attached to the final response.

    Produced by the citation node and carried in ``OutputState.citations``
    (and embedded into the AIMessage ``additional_kwargs`` by
    ``output_adapter``).
    """

    id: str
    programme_id: str
    programme_name: str
    section: str
    source_type: str
    content: str
    confidence: str
    url: str | None

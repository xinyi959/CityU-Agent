"""Metadata retriever: vector retrieval + structured resolution (unified format).

Exact facts (tuition fee, deadline, duration, ...) are NOT semantic
knowledge, so when the query identifies a programme we resolve it directly
against ``data/programmes.json`` and return the exact structured value --
no embedding involved.

Output (unified tool format):

    resolved -> {type: "metadata", programme_id, field, value}
                value = raw structured field (e.g. tuition_fee dict with
                local + non_local; default returns both)

    fallback -> [ {type: "metadata", programme_id, field, value: text}, ... ]
                semantic search on the ``programme_metadata`` collection.
"""

from rag.programme_resolver import extract_field, extract_programme_ref, find_programme
from rag.retriever import retrieve_metadata, to_metadata


def metadata_retriever_node(state):
    query = state["query"]

    ref = extract_programme_ref(query)
    field = extract_field(query)
    programme = find_programme(ref)

    if programme is not None:
        metadata = programme.get("metadata", {})
        value = metadata.get(field) if field else metadata
        return {
            "documents": [
                {
                    "type": "metadata",
                    "programme_id": programme["programme_id"],
                    "field": field,
                    "value": value,
                }
            ],
            "programme_id": programme["programme_id"],
            "programme_name": programme.get("name"),
        }

    # fallback: no resolvable programme -> semantic search on metadata index
    return {
        "documents": [to_metadata(d) for d in retrieve_metadata(query)]
    }

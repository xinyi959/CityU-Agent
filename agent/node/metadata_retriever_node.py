"""Metadata retriever: structured resolution + vector fallback (Evidence).

Exact facts (tuition fee, deadline, duration, ...) are NOT semantic
knowledge, so when the query identifies a programme we resolve it directly
against ``data/programmes.json`` and return the exact structured value.

Resolved -> single Evidence with score 1.0 (exact match).
Fallback -> Evidence objects from ``programme_metadata`` vector search.
"""

from rag.evidence import Evidence
from rag.metadata_builder import build_metadata_document, value_to_text
from rag.programme_resolver import extract_field, extract_programme_ref, find_programme
from rag.retriever import retrieve_metadata


def metadata_retriever_node(state):
    query = state["query"]

    ref = extract_programme_ref(query)
    field = extract_field(query)
    programme = find_programme(ref)

    if programme is not None:
        if field:
            content = value_to_text(programme.get("metadata", {}).get(field)) or ""
        else:
            content = build_metadata_document(programme).page_content
        evidence = [
            Evidence(
                programme_id=programme["programme_id"],
                section=field or "metadata",
                content=content,
                score=1.0,
            )
        ]
        return {
            "evidence": evidence,
            "programme_id": programme["programme_id"],
            "programme_name": programme.get("name"),
        }

    # fallback: no resolvable programme -> semantic search on metadata index
    evidence = [
        Evidence(
            programme_id=doc.metadata["programme_id"],
            section=doc.metadata.get("field") or "metadata",
            content=doc.page_content,
            score=score,
        )
        for doc, score in retrieve_metadata(query)
    ]
    return {
        "evidence": evidence
    }

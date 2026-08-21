from rag.evidence import Evidence
from rag.retriever import retrieve_section
from rag.programme_resolver import resolve_programme_ref


def section_retriever_node(state):

    query = state["query"]

    # Resolve the programme with the same fallback chain as the metadata
    # retriever: router ref -> previously confirmed ref -> text rules on the
    # current query -> text rules on recent human messages. Keeps an omitted
    # referent ("what about English requirement?") from degrading the search
    # to the whole section corpus.
    programme = resolve_programme_ref(
        query,
        programme_ref=state.get("programme_ref"),
        resolved_ref=state.get("resolved_programme_ref"),
        messages=state.get("messages", []),
    )

    programme_id = programme["programme_id"] if programme else None

    # Recommendation-scoped path: no single programme resolved but an earlier
    # summary decision in the same turn produced a set of programme ids ->
    # filter the section search to them.
    scope_ids = state.get("programme_ids") or []

    if programme_id:
        docs = retrieve_section(query, programme_id=programme_id, k=5)
    elif scope_ids:
        docs = retrieve_section(query, programme_ids=scope_ids, k=5)
    else:
        docs = retrieve_section(query, k=5)

    evidence = [
        Evidence(
            id=f"{doc.metadata['programme_id']}-{doc.metadata['section']}",
            programme_id=doc.metadata["programme_id"],
            section=doc.metadata["section"],
            content=doc.page_content,
            score=score,
            source_type="retrieval"
        )
        for doc, score in docs
    ]

    out = {
        "evidence": evidence,
    }
    if programme:
        out["resolved_programme_ref"] = {
            "programme_id": programme["programme_id"],
            "programme_name": programme.get("name"),
        }
    return out

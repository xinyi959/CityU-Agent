from rag.evidence import Evidence
from rag.retriever import retrieve_section
from rag.programme_resolver import resolve_programme_scope


def section_retriever_node(state):

    query = state["query"]

    # Resolve the programme(s) to scope to: explicit query mention -> scope
    # set (this turn's recommendation / previous turn's persisted set) ->
    # router inferred ref -> message history.
    programmes = resolve_programme_scope(
        query,
        programme_ref=state.get("programme_ref"),
        messages=state.get("messages", []),
        scope_ids=state.get("programme_ids") or [],
    )

    ids = [p["programme_id"] for p in programmes]

    if len(ids) == 1:
        docs = retrieve_section(query, programme_id=ids[0], k=5)
    elif len(ids) > 1:
        docs = retrieve_section(query, programme_ids=ids, k=5)
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
    if programmes:
        # Persist the scope so the next omitted referent keeps the same set.
        out["resolved_programme_refs"] = [
            {
                "programme_id": p["programme_id"],
                "programme_name": p.get("name"),
            }
            for p in programmes
        ]
    return out

from rag.evidence import Evidence
from rag.retriever import retrieve_section
from rag.programme_resolver import find_programme

def section_retriever_node(state):

    query = state["query"]

    programme_id = None
    programme_ref = state.get(
        "programme_ref"
    )

    print("PROGRAMME REF DEBUG:", programme_ref, type(programme_ref))
    if programme_ref:

        programme = find_programme(
            programme_ref
        )

        if programme:
            programme_id = programme["programme_id"]


    docs = retrieve_section(
        query,
        programme_id=programme_id,
        k=5
    )


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


    return {
        "evidence": evidence,
        "retrieval_type": "section",
    }

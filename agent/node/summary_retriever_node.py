from rag.evidence import Evidence
from rag.retriever import retrieve_summary


def summary_retriever_node(state):
    """Recommendation path: whole-programme summaries as Evidence objects."""
    evidence = [
        Evidence(
            id=f"{doc.metadata['programme_id']}-summary",
            programme_id=doc.metadata["programme_id"],
            section="Programme Summary",
            content=doc.page_content,
            score=score,
        )
        for doc, score in retrieve_summary(state["query"], k=5)
    ]
    return {
        "evidence": evidence
    }

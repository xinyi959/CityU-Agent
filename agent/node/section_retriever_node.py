from rag.evidence import Evidence
from rag.retriever import retrieve_section


def section_retriever_node(state):
    """Detailed-QA path: programme section documents as Evidence objects."""
    evidence = [
        Evidence(
            programme_id=doc.metadata["programme_id"],
            section=doc.metadata["section"],
            content=doc.page_content,
            score=score,
        )
        for doc, score in retrieve_section(state["query"], k=5)
    ]
    return {
        "evidence": evidence
    }

from rag.retriever import retrieve_section, to_section


def section_retriever_node(state):
    """QA path: retrieve programme section documents (unified format)."""
    docs = retrieve_section(state["query"], k=5)
    return {
        "documents": [to_section(d) for d in docs]
    }

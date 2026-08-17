from rag.retriever import retrieve_section


def section_retriever_node(state):
    """QA path: retrieve programme section documents."""
    docs = retrieve_section(state["query"], k=5)
    return {
        "documents": docs
    }

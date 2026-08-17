from rag.retriever import retrieve_metadata


def metadata_retriever_node(state):
    """Exact-fact path: retrieve structured metadata documents."""
    docs = retrieve_metadata(state["query"], k=5)
    return {
        "documents": docs
    }

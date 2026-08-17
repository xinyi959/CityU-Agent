from rag.retriever import retrieve_summary


def summary_retriever_node(state):
    """Recommendation path: retrieve whole-programme summaries."""
    docs = retrieve_summary(state["query"], k=5)
    return {
        "documents": docs
    }

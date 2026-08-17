from rag.retriever import retrieve_summary, to_summary


def summary_retriever_node(state):
    """Recommendation path: retrieve whole-programme summaries (unified format)."""
    docs = retrieve_summary(state["query"], k=5)
    return {
        "documents": [to_summary(d) for d in docs]
    }

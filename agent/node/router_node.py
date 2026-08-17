from rag.router import classify_query


def router_node(state):

    query = state["query"]

    intent = classify_query(query)

    return {
        "intent": intent
    }
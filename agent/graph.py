"""CityUHK postgraduate assistant -- rule-based RAG graph (v1).

Graph:

    START
      |
      v
    router  (classify_query: summary | metadata | section)
      |
      |-- summary  --> summary_retriever --+
      |-- metadata --> metadata_retriever -+--+
      |                                    |  |
      `-- section --> section_retriever ---+  |
                                                v
                                              answer
                                                |
                                                v
                                               END

* summary  -> whole-programme summaries (programme_summaries)
* metadata -> structured metadata facts (programme_metadata)
* section  -> programme section documents (programme_sections)
"""

from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from agent.node.answer_node import answer_node
from agent.node.metadata_retriever_node import metadata_retriever_node
from agent.node.router_node import router_node
from agent.node.section_retriever_node import section_retriever_node
from agent.node.summary_retriever_node import summary_retriever_node

load_dotenv()


# ==================================
# State
# ==================================


class AgentState(TypedDict):
    query: str
    intent: str
    documents: list
    answer: str
    # populated by metadata_retriever when the query resolves to a programme
    programme_id: str
    programme_name: str


# ==================================
# Graph
# ==================================


def build_graph(answer_node=answer_node):
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("summary_retriever", summary_retriever_node)
    graph.add_node("metadata_retriever", metadata_retriever_node)
    graph.add_node("section_retriever", section_retriever_node)
    graph.add_node("answer", answer_node)

    # START -> router
    graph.add_edge(START, "router")

    # router -> retriever by intent
    graph.add_conditional_edges(
        "router",
        lambda state: state["intent"],
        {
            "summary": "summary_retriever",
            "metadata": "metadata_retriever",
            "section": "section_retriever",
        },
    )

    # all retrievers -> answer -> END
    graph.add_edge("summary_retriever", "answer")
    graph.add_edge("metadata_retriever", "answer")
    graph.add_edge("section_retriever", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


app = build_graph()


# ==================================
# Demo (graph structure only -- no LLM call)
# ==================================

if __name__ == "__main__":
    print("nodes:", list(app.get_graph().nodes.keys()))
    print("edges:")
    for u, v, data in app.get_graph().edges:
        print(f"  {u} -> {v} {data}")

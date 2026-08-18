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
                                              generator
                                                |
                                                v
                                              citation
                                                |
                                                v
                                               END

* summary  -> whole-programme summaries (programme_summaries)
* metadata -> structured metadata facts (programme_metadata)
* section  -> programme section documents (programme_sections)
* generator generates the answer from Evidence; citation appends Sources.
"""

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage

from agent.nodes.answer_node import generate_answer
from agent.nodes.citation import citation_formatter
from agent.nodes.metadata_retriever_node import metadata_retriever_node
from agent.nodes.router_node import router_node
from agent.nodes.section_retriever_node import section_retriever_node
from agent.nodes.summary_retriever_node import summary_retriever_node

from agent.state import AgentState

load_dotenv()

def input_adapter(state: AgentState):
    messages = state.get("messages", [])

    if not messages:
        raise ValueError("No messages provided")

    while (
        isinstance(messages, list)
        and len(messages) == 1
        and isinstance(messages[0], list)
    ):
        messages = messages[0]

    last_message = messages[-1]

    # LangChain Message
    if hasattr(last_message, "content"):
        content = last_message.content

    # dict message
    elif isinstance(last_message, dict):
        content = last_message.get("content")

    else:
        raise TypeError(
            f"Unsupported message type: {type(last_message)}"
        )


    # content block format:
    #
    # [
    #   {
    #      "type": "text",
    #      "text": "hello"
    #   }
    # ]
    if isinstance(content, list):
        texts = []

        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
            else:
                texts.append(str(block))

        query = "".join(texts)

    elif isinstance(content, str):
        query = content

    else:
        raise TypeError(
            f"Unsupported content type: {type(content)}: {content}"
        )

    return {
        "query": query
    }

def output_adapter(state: AgentState):

    return {
        "messages": [
            AIMessage(
                content=state["final_response"],
                additional_kwargs={
                    "citations": state.get("citations", [])
                }
            )
        ]
    }

# ==================================
# Graph
# ==================================


def build_graph(answer_node=generate_answer, citation_node=citation_formatter):
    graph = StateGraph(AgentState)

    graph.add_node("input_adapter", input_adapter)
    graph.add_node("output_adapter", output_adapter)

    graph.add_node("router", router_node)
    graph.add_node("summary_retriever", summary_retriever_node)
    graph.add_node("metadata_retriever", metadata_retriever_node)
    graph.add_node("section_retriever", section_retriever_node)

    graph.add_node("generator", answer_node)
    graph.add_node("citation", citation_node)

    # START -> router
    graph.add_edge(START, "input_adapter")
    graph.add_edge("input_adapter", "router")

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

    # all retrievers -> generator -> citation -> END
    graph.add_edge("summary_retriever", "generator")
    graph.add_edge("metadata_retriever", "generator")
    graph.add_edge("section_retriever", "generator")

    # generator -> citation -> output_adapter
    graph.add_edge("generator", "citation")
    graph.add_edge("citation", "output_adapter")

    # output_adapter -> END
    graph.add_edge("output_adapter", END)

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

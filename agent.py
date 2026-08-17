from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter
from langchain.tools import tool
from langchain_core.messages import AnyMessage

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

from langgraph.graph.message import add_messages


# ==================================
# 1. Load environment variables
# ==================================

load_dotenv()


# ==================================
# 2. Define Tool
# ==================================
from rag.retriever import search_programmes
@tool
def search_cityu_programmes(query: str) -> str:
    """
    Search City University of Hong Kong taught postgraduate programmes database.

    Use this tool when the user asks about:
    - available programmes
    - admission requirements
    - english requirements
    - tuition fees
    - programme structure
    - course information

    The database contains official Markdown
    pages extracted from CityUHK SGS website.

    Input:
        query:
        A semantic search query describing
        the user's information need.

    Output:
        Relevant programme information.
    """
    print("\n========== TOOL CALLED ==========")
    print("Query:")
    print(query)

    result = search_programmes(query)

    print("\n========== TOOL RESULT ==========")

    print(result[:2000])

    return result


tools = [
    search_cityu_programmes
]


# ==================================
# 3. Initialize OpenRouter LLM
# ==================================

model = ChatOpenRouter(
    model="deepseek/deepseek-v4-flash",
    temperature=0,
)


model_with_tools = model.bind_tools(
    tools
)

from langchain_core.messages import SystemMessage
SYSTEM_PROMPT = """
You are a CityUHK taught postgraduate programme advisor.

Your task is to help prospective students find and understand suitable master's programmes at City University of Hong Kong.

Rules:
- Use the search tool whenever programme information is needed.
- Answer only based on the retrieved programme data.
- Do not make up programme details, requirements, fees, or courses.
- When recommending programmes, explain the match based on the user's background and the retrieved information.
- Be concise, clear, and practical.
"""

# ==================================
# 4. Define LangGraph State
# ==================================

class AgentState(TypedDict):

    messages: Annotated[
        list[AnyMessage],
        add_messages
    ]



# ==================================
# 5. Define Agent Node
# ==================================

def agent_node(state):

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        )
    ] + state["messages"]


    response = model_with_tools.invoke(
        messages
    )


    return {
        "messages":[response]
    }



# ==================================
# 6. Define Tool Node
# ==================================

tool_node = ToolNode(
    tools
)



# ==================================
# 7. Build Graph
# ==================================

graph = StateGraph(
    AgentState
)


# add nodes

graph.add_node(
    "agent",
    agent_node
)


graph.add_node(
    "tools",
    tool_node
)



# START -> agent

graph.add_edge(
    START,
    "agent"
)



# agent decides:
#
# tool call?
#
# yes -> tools
# no  -> END

graph.add_conditional_edges(
    "agent",
    tools_condition
)



# after tool execution:
#
# tools -> agent

graph.add_edge(
    "tools",
    "agent"
)



# compile

app = graph.compile()



# ==================================
# 8. Run Test
# ==================================

if __name__ == "__main__":


    question = """
    I want to apply some AI related program. I used to study computer science.By the way,How much is the tuition fee for local students to apply for the computer science program?
    """


    result = app.invoke(
        {
            "messages": [
                (
                    "user",
                    question
                )
            ]
        },
        stream_mode="values"
    )


    print("\n========== FINAL ANSWER ==========\n")


    print(
        result["messages"][-1].content
    )
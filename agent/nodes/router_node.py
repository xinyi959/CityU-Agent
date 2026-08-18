from agent.state.router_schema import RouterDecision
from langchain_core.messages import SystemMessage
from agent.llm import model

ROUTER_PROMPT = """
You are the routing agent for CityUHK postgraduate assistant.

Your task has two steps:

1. Identify user's intent.
2. Decide which retrieval source should be used.


## Intent

intent describes user's goal.

Options:

qa:
- asking factual programme information
- asking about requirements, fees, duration, details

recommendation:
- asking which programme suits them
- asking for suggestions

comparison:
- comparing multiple programmes


## Retrieval Type

retrieval_type decides which retriever should handle the request.

metadata:
- exact structured facts
- tuition fee
- deadline
- duration
- credit
- study mode

section:
- detailed programme information
- entrance requirements
- curriculum
- course information

summary:
- programme recommendation
- programme overview
- programme comparison


Return structured JSON only.
"""

router_llm = model.with_structured_output(
    RouterDecision
)

def router_node(state):

    messages = state["messages"]

    decision = router_llm.invoke(
        [
            SystemMessage(
                content=ROUTER_PROMPT
            ),
            *messages
        ]
    )
    print("ROUTER DECISION:")
    print(decision)


    return {
        "intent": decision.intent,
        "retrieval_type": decision.retrieval_type,
        "field": decision.field,
        "programme_id": decision.programme_id,
    }
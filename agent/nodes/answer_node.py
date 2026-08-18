from langchain_core.messages import HumanMessage, SystemMessage
from agent.llm import model
from rag.evidence import Evidence

QA_PROMPT = """
You answer CityUHK postgraduate programme questions.

Rules:

- Answer EVERY sub-question in the user query, in the order asked.
- Match each sub-question to the evidence block that covers it
  (evidence blocks are labeled [programme | section]).
- For exact facts (tuition fee, deadline, duration, credits, mode of
  study): state the value directly, use bullet lists, do not explain
  beyond the provided value.
- For detailed content (entrance requirements, curriculum, courses):
  preserve important conditions and requirements, use bullet lists,
  do not summarize away critical details.
- If the evidence does not cover a sub-question, say so explicitly
  instead of guessing.
- Do not invent facts that are not in the evidence.
"""

SUMMARY_PROMPT = """
You recommend CityUHK programmes.

Rules:

- Explain why each programme matches.
- Compare programmes when multiple options exist.
- Mention programme names clearly.

Format:

Programme:

Why it fits:
- ...
"""


def format_evidence(evidence_list) -> str:
    """Render Evidence objects as LLM context blocks."""
    blocks = []
    for item in evidence_list:
        if isinstance(item, Evidence):
            blocks.append(item.render())
        else:
            blocks.append(str(item))
    return "\n\n".join(blocks)


def generate_answer(state):
    print(
        "RETRIEVAL TYPE:",
        state.get("retrieval_type"),
        "INTENT:",
        state.get("intent"),
    )

    intent = state.get("intent")
    prompt = (
        SUMMARY_PROMPT
        if intent in ("recommendation", "comparison")
        else QA_PROMPT
    )

    context = format_evidence(state["evidence"])

    messages = [
        SystemMessage(
            content=prompt
        ),
        HumanMessage(
            content=(
                f"User query:\n{state['query']}\n\n"
                f"Retrieved context:\n{context}"
            )
        ),
    ]
    response = model.invoke(messages)
    return {
        "answer": response.content
    }


# backward-compatible alias
answer_node = generate_answer

from langchain_core.messages import HumanMessage, SystemMessage
from agent.llm import model
from rag.evidence import Evidence

METADATA_PROMPT = """
You answer exact programme facts.

Rules:

- Give the answer directly.
- Always use bullet lists for structured fields.
- Do not explain beyond the provided value.

Format:

Answer:

- Field: Value
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

SECTION_PROMPT = """
You answer detailed programme questions.

Rules:

- Extract relevant information only.
- Preserve important requirements.
- Use bullet lists for multiple requirements.
- Do not summarize away critical conditions.
"""

PROMPTS = {
    "metadata": METADATA_PROMPT,
    "section": SECTION_PROMPT,
    "summary": SUMMARY_PROMPT,
}

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
        state.get("retrieval_type")
    )
    context = format_evidence(state["evidence"])

    prompt = PROMPTS.get(
        state.get("retrieval_type")
    )
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

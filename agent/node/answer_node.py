from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

load_dotenv()

ANSWER_PROMPT = """You are a CityUHK postgraduate assistant.

The retrieved context may come from:

1. Programme summary documents:
   used for programme recommendation.

2. Programme metadata documents:
   used for exact factual fields (fees, deadlines, study periods, modes).

3. Programme section documents:
   used for factual answers.

Use only provided context.

If recommending programmes:
- explain why each programme fits the user's background.
- mention programme names clearly.

If answering factual questions:
- directly answer from the relevant section."""

model = ChatOpenRouter(
    model="deepseek/deepseek-v4-flash",
    temperature=0,
)


def format_documents(documents) -> str:
    """Render unified tool-output dicts (or legacy Documents) as context."""
    blocks = []
    for item in documents:
        if isinstance(item, dict):
            blocks.append(_format_dict(item))
        else:  # legacy langchain Document
            m = item.metadata
            label = m.get("section") or m.get("type", "document")
            blocks.append(f"[{m.get('programme_id')} | {label}]\n{item.page_content}")
    return "\n\n".join(blocks)


def _format_dict(item: dict) -> str:
    t = item.get("type", "?")
    pid = item.get("programme_id", "?")
    if t == "summary":
        return f"[{pid} | summary] {item.get('name')}\n{item.get('context', '')}"
    if t == "metadata":
        return (
            f"[{pid} | metadata: {item.get('field') or 'all fields'}]\n"
            f"{item.get('value', '')}"
        )
    return f"[{pid} | {item.get('section')}]\n{item.get('context', '')}"


def answer_node(state):
    context = format_documents(state["documents"])
    messages = [
        SystemMessage(content=ANSWER_PROMPT),
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

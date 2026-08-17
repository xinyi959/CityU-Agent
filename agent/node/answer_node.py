from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

from rag.evidence import Evidence

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


def format_evidence(evidence_list) -> str:
    """Render Evidence objects (or legacy dicts) as LLM context blocks."""
    blocks = []
    for item in evidence_list:
        if isinstance(item, Evidence):
            blocks.append(item.render())
        elif isinstance(item, dict):
            t = item.get("type", "?")
            pid = item.get("programme_id", "?")
            if t == "metadata":
                blocks.append(
                    f"[{pid} | metadata: {item.get('field')}]\n{item.get('value', '')}"
                )
            else:
                blocks.append(
                    f"[{pid} | {item.get('section')}]\n"
                    f"{item.get('context', item.get('value', ''))}"
                )
        else:
            blocks.append(str(item))
    return "\n\n".join(blocks)


def generate_answer(state):
    context = format_evidence(state["evidence"])
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


# backward-compatible alias
answer_node = generate_answer

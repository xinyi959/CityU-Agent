from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

load_dotenv()

ANSWER_PROMPT = """You are a CityUHK postgraduate assistant.

The retrieved context may come from:

1. Programme summary documents:
   used for programme recommendation.

2. Programme section documents:
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
    blocks = []
    for doc in documents:
        m = doc.metadata
        label = m.get("section") or m.get("type", "document")
        blocks.append(
            f"[{m.get('programme_id')} | {label}]\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


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

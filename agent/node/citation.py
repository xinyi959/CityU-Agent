"""Citation formatter: builds the final response with id-anchored sources.

Runs AFTER the generator. Adds a "Sources:" block where every entry is
anchored by the Evidence id (easy to debug) and shows the programme name,
section and confidence:

    <answer>

    Sources:

    [P53-tuition_fee]
    MSc Computer Science > Tuition Fee
    Confidence: 1.0

The structured ``citations`` list (id, programme, section, score, content)
is stored in state for programmatic use.
"""

from rag.programme_resolver import get_programmes


def _name_map() -> dict:
    return {p["programme_id"]: p.get("name") for p in get_programmes()}


def citation_formatter(state):
    name_map = _name_map()

    citations = []
    for e in state["evidence"]:
        citations.append(
            {
                **e.to_citation(),
                "programme_name": name_map.get(e.programme_id, e.programme_id),
                "content": e.content,
            }
        )

    final_response = f"{state['answer']}\n\nSources:\n"
    for c in citations:
        final_response += (
            f"\n[{c['id']}]\n"
            f"{c['programme_name']} > {c['section']}\n"
            f"Confidence: {c['score']}"
        )

    return {
        "citations": citations,
        "final_response": state["answer"]
    }

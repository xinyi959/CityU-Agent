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
from agent.models import Citation
from rag.programme_resolver import get_programmes


def _name_map() -> dict:
    return {p["programme_id"]: p.get("name") for p in get_programmes()}

def _confidence(evidence):

    if evidence.source_type == "metadata":
        return "High"

    score = evidence.score

    if score < 0.3:
        return "High"

    if score < 0.5:
        return "Medium"

    return "Low"


def citation_formatter(state):

    name_map = _name_map()

    citations: list[Citation] = []

    for e in state["evidence"]:

        citations.append(
            {
                "id": e.id,
                "programme_id": e.programme_id,
                "programme_name": (
                    name_map.get(
                        e.programme_id,
                        e.programme_id
                    )
                ),
                "section": e.section,
                "source_type": e.source_type,
                "content": e.content,
                "confidence": _confidence(e),
                "url": (
                    e.metadata.get("url")
                    if e.metadata
                    else None
                ),
            }
        )


    return {
        "citations": citations,
        "final_response": state["answer"],
    }

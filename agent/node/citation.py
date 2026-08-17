"""Citation formatter: builds the final response with evidence sources.

Runs AFTER the generator: turns the answer + evidence list into a
``final_response`` with a numbered "Sources:" block, and stores the
structured ``citations`` list in state.
"""


def citation_formatter(state):
    citations = []
    for e in state["evidence"]:
        citations.append(
            {
                "programme": e.programme_id,
                "section": e.section,
                "score": round(e.score, 2),
            }
        )

    final_response = f"{state['answer']}\n\nSources:\n"

    for i, c in enumerate(citations, 1):
        final_response += (
            f"\n[{i}] "
            f"{c['programme']} > "
            f"{c['section']}"
        )

    return {
        "citations": citations,
        "final_response": final_response
    }

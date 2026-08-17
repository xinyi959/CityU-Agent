"""Citation formatter: builds the final response with evidence sources + content.

Runs AFTER the generator: turns the answer + evidence list into a
``final_response`` with a numbered "[Sources]" block and a "[Evidence]"
block quoting each supporting snippet, and stores the structured
``citations`` list (with content) in state.

    <answer>

    [Sources]

    1. P53 > Tuition Fee

    [Evidence]

    "Local Students: HK$7,600 per credit
     Non-local Students: HK$9,100 per credit"
"""

EVIDENCE_SNIPPET_CHARS = 300


def _display_section(section: str) -> str:
    """'tuition_fee' -> 'Tuition Fee'; leave plain titles untouched."""
    if "_" in section:
        return section.replace("_", " ").title()
    return section


def citation_formatter(state):
    citations = []
    for e in state["evidence"]:
        citations.append(
            {
                "programme": e.programme_id,
                "section": e.section,
                "score": round(e.score, 2),
                "content": e.content,
            }
        )

    final_response = f"{state['answer']}\n\n[Sources]\n"

    for i, c in enumerate(citations, 1):
        final_response += (
            f"\n{i}. {c['programme']} > "
            f"{_display_section(c['section'])}"
        )

    final_response += "\n\n[Evidence]\n"
    for c in citations:
        snippet = c["content"][:EVIDENCE_SNIPPET_CHARS]
        final_response += f'\n"{snippet}"\n'

    return {
        "citations": citations,
        "final_response": final_response
    }

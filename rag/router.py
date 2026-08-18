"""Rule-based intent router for the CityUHK programme assistant (v2).

Three knowledge layers, three retrieval paths:

  * metadata -- exact structured facts (fee, tuition, deadline, duration,
                credit, mode of study ...)        -> programme_metadata index
  * summary  -- programme recommendation (recommend, suggest, suitable,
                which programme, choose ...)      -> programme_summaries index
  * section  -- detailed QA (requirements, courses, content ...)
                                                  -> programme_sections index

Priority: metadata keywords are checked FIRST (factual fields are the
strongest signal), then summary keywords, everything else -> section.
"""

import re

METADATA_KEYWORDS = [
    "fee",
    "tuition",
    "deadline",
    "duration",
    "credit",
    "mode",
]

SUMMARY_KEYWORDS = [
    "recommend",
    "suggest",
    "suitable",
    "which programme",
    "best programme",
    "what should i study",
    "choose",
    "apply for"
]


def _has_keyword(q: str, keyword: str) -> bool:
    """Match a keyword against the query.

    Single-word keywords use a word-boundary guard (\bkw\w*) so "fee" matches
    "fee"/"fees" but not "coffee"/"feedback", and "mode" matches "mode" but
    not "model"/"modern". Multi-word phrases fall back to a plain substring.
    """
    if " " not in keyword:
        return bool(re.search(rf"\b{re.escape(keyword)}\w*", q))
    return keyword in q


def classify_query(query: str) -> str:
    """Return 'metadata', 'summary' or 'section' for the given user query."""
    q = query.lower()

    for keyword in METADATA_KEYWORDS:
        if _has_keyword(q, keyword):
            return "metadata"

    for keyword in SUMMARY_KEYWORDS:
        if _has_keyword(q, keyword):
            return "summary"

    return "section"


if __name__ == "__main__":
    examples = [
        "What is the tuition fee of MSc Mechanical Engineering?",
        "When is the application deadline for P02?",
        "How many credits are required for this programme?",
        "What is the normal study duration?",
        "Is this programme professionally accredited?",
        "I am interested in AI, recommend programmes",
        "Which programme is suitable for an engineering graduate?",
        "What should I study if I like data analysis?",
        "What are the entrance requirements of MA International Accounting?",
        "Which programmes require CET-6 450?",
        "Tell me about CityU",
    ]
    for ex in examples:
        print(f"{classify_query(ex):<10} <- {ex}")

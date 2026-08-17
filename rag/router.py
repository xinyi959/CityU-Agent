"""Rule-based intent router for the CityUHK programme assistant (v1).

The domain is narrow (taught postgraduate programmes at CityUHK), so user
intent is easy to classify with keywords:

  * recommendation -- "recommend / suggest / which programme / suitable /
                      best programme / what should I study / help me choose"
  * qa             -- factual questions about fee / tuition / requirements /
                      IELTS / TOEFL / deadline / course / credit / duration

Strategy: recommendation keywords are checked first (they are stronger
signals), then qa keywords; anything unmatched defaults to ``qa``.
"""

RECOMMENDATION_KEYWORDS = [
    "recommend",
    "suggest",
    "suitable",
    "which programme",
    "best programme",
    "what should i study",
    "choose",
]

QA_KEYWORDS = [
    "fee",
    "tuition",
    "requirement",
    "ielts",
    "toefl",
    "deadline",
    "course",
    "credit",
    "duration",
]


def classify_query(query: str) -> str:
    """Return 'recommendation' or 'qa' for the given user query."""
    q = query.lower()

    for keyword in RECOMMENDATION_KEYWORDS:
        if keyword in q:
            return "recommendation"

    for keyword in QA_KEYWORDS:
        if keyword in q:
            return "qa"

    return "qa"


if __name__ == "__main__":
    examples = [
        "I am interested in AI, recommend programmes",
        "Which programme is suitable for an engineering graduate?",
        "What should I study if I like data analysis?",
        "Help me choose a master's programme in finance",
        "What is the tuition fee of MA International Accounting?",
        "IELTS requirement for P66",
        "When is the application deadline?",
        "How many credits are required?",
        "What is the normal study duration?",
        "Tell me about CityU",
    ]
    for ex in examples:
        print(f"{classify_query(ex):<14} <- {ex}")

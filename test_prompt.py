from agent.graph import app
from langchain_core.messages import HumanMessage

TEST_CASES = [
    {
        "name": "Metadata - Tuition Fee",
        "query": "What is the tuition fee of MSc Computer Science?",
    },
    {
        "name": "Section - Entrance Requirement",
        "query": "What are the entrance requirements of MSc Artificial Intelligence?",
    },
    {
        "name": "Summary - Programme Recommendation",
        "query": (
            "I studied Computer Science. "
            "Which CityUHK postgraduate programme should I choose?"
        ),
    },
]


def run_test(query: str):

    print("=" * 80)
    print("QUERY:")
    print(query)
    print("-" * 80)

    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content=query
                )
            ]
        }
    )

    print("\nRETRIEVAL TYPE:")
    print(result.get("retrieval_type"))

    print("\nANSWER:")
    print(result.get("answer"))

    print("\nCITATIONS:")
    for c in result.get("citations", []):
        print(
            f"- {c.get('programme_name')} | "
            f"{c.get('section')} | "
            f"{c.get('source_type')}"
        )

    print()


if __name__ == "__main__":

    for case in TEST_CASES:

        print("\nTEST:")
        print(case["name"])

        run_test(
            case["query"]
        )
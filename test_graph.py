"""
Test CityU-Agent graph directly.

Run from project root:

    uv run python test_graph.py
"""

from agent.graph import app

print("GRAPH NODES:")
print(list(app.get_graph().nodes.keys()))

print("\nGRAPH EDGES:")
for edge in app.get_graph().edges:
    print(edge)

def main():
    query = "How much should I pay for MSc Computer Science?"

    print("=" * 80)
    print("Testing CityU-Agent graph")
    print("=" * 80)
    print(f"Query: {query}")
    print()

    # Test with the same message-style input expected by a chat UI.
    input_state = {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ]
    }

    print("Input state:")
    print(input_state)
    print()

    try:
        result = app.invoke(input_state)
        print("*" * 80)
        print(result["messages"][-1])

        print("=" * 80)
        print("GRAPH RESULT")
        print("=" * 80)

        print(result)

        print()
        print("=" * 80)
        print("FINAL RESPONSE")
        print("=" * 80)

        print(result.get("final_response", "<no final_response>"))

        print()
        print("=" * 80)
        print("MESSAGES")
        print("=" * 80)

        for message in result.get("messages", []):
            print(message)

    except Exception as e:
        print()
        print("=" * 80)
        print("GRAPH FAILED")
        print("=" * 80)

        print(f"{type(e).__name__}: {e}")

        raise


if __name__ == "__main__":
    main()
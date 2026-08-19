"""Interactive CLI for the CityUHK programme agent (real LLM calls).

Usage:
    one-shot:  .venv/bin/python -m agent.cli "What is the tuition fee of MSc Mechanical Engineering?"
    repl:      .venv/bin/python -m agent.cli          (type queries, Ctrl+D / 'exit' to quit)
"""

from __future__ import annotations

import sys
import time

from agent.graph import app


def answer(query: str) -> tuple[dict, float]:
    """Run the graph and return its final full state.

    ``app.invoke`` returns only the public OutputState (messages,
    final_response, citations); the CLI is a debug tool, so it streams the
    internal values to also show routing intent and evidence counts.
    """
    t0 = time.time()
    result = None
    for chunk in app.stream({"query": query}, stream_mode="values"):
        result = chunk
    return result, time.time() - t0


def pretty_print(result: dict, elapsed: float) -> None:
    print(f"[intent] {result['intent']}"
          f"  |  evidence {len(result['evidence'])}  |  {elapsed:.1f}s")
    print(f"\n[answer]\n\n{result['final_response']}\n")


def repl() -> None:
    print("CityUHK postgraduate assistant — type a question ('exit' to quit)\n")
    while True:
        try:
            query = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("bye")
            return
        try:
            result, elapsed = answer(query)
            pretty_print(result, elapsed)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {type(exc).__name__}: {exc}")


def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        try:
            result, elapsed = answer(query)
            pretty_print(result, elapsed)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {type(exc).__name__}: {exc}")
        return
    repl()


if __name__ == "__main__":
    main()

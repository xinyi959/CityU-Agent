"""Interactive CLI for the CityUHK programme agent (real LLM calls).

Usage:
    one-shot:  .venv/bin/python -m agent.cli "What is the tuition fee of MSc Mechanical Engineering?"
    repl:      .venv/bin/python -m agent.cli          (type queries, Ctrl+D / 'exit' to quit)
"""

from __future__ import annotations

import sys
import time

from agent import app


def answer(query: str) -> dict:
    t0 = time.time()
    result = app.invoke({"query": query})
    return result, time.time() - t0


def pretty_print(result: dict, elapsed: float) -> None:
    print(f"\n[intent] {result['intent']}"
          f"  |  retrieved {len(result['documents'])} doc(s)"
          f"  |  {elapsed:.1f}s")
    for item in result["documents"]:
        label = item.get("name") or item.get("section") or item.get("field") or "?"
        print(f"  - [{item['type']}] {item['programme_id']} | {label}")
    print(f"\n[answer]\n{result['answer']}\n")


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

"""Evidence object -- the unit flowing from retrievers to the generator.

Replaces raw Documents/dicts in the agent state so the answer and citation
layers can reference each retrieved item precisely.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Evidence:
    programme_id: str
    section: str
    content: str
    score: float

    def render(self) -> str:
        """Context block for the LLM prompt."""
        return f"[{self.programme_id} | {self.section}]\n{self.content}"

    def to_citation(self) -> dict:
        return {
            "programme": self.programme_id,
            "section": self.section,
            "score": round(self.score, 2),
        }

"""Evidence object -- the unit flowing from retrievers to the generator.

Replaces raw Documents/dicts in the agent state so the answer and citation
layers can reference each retrieved item precisely.

    id         -- stable anchor, e.g. "P53-tuition_fee" (for citations/debug)
    section    -- display name, e.g. "Tuition Fee" / "Entrance Requirements"
    content    -- the text the LLM needs (URLs and other non-LLM details are
                  kept OUT of content)
    score      -- retrieval confidence (1.0 = exact structured match)
    metadata   -- extra fields for the formatter, e.g. {"url": "..."}
"""

from __future__ import annotations

from dataclasses import dataclass, field

# RAG内部对象，给模型看的
@dataclass
class Evidence:
    id: str
    programme_id: str
    section: str
    content: str
    score: float
    source_type: str = "retrieval"
    metadata: dict | None = field(default=None)

    def render(self) -> str:
        """Context block for the LLM prompt (content only, no metadata)."""
        return f"[{self.programme_id} | {self.section}]\n{self.content}"

    def to_citation(self) -> dict:
        return {
            "id": self.id,
            "programme_id": self.programme_id,
            "section": self.section,
            "score": round(self.score, 2),
        }

"""Build programme summary documents for the recommendation use case.

Input : ``data/programmes.json`` (output of ``rag/parser.py``)
Output: ``List[Document]`` -- one summary document per programme (64 docs),
        for the Chroma collection ``programme_summaries``.

Rule-based extraction (v1, no LLM):

    summary_text = f\"\"\"
    Programme:
    {name}

    Overview:
    {Programme Aims and Objectives}

    Courses:
    {Course Description | Programme Content, first 500 chars}

    Highlights:
    {Professional Accreditation / Career / Bonus Features / ... , first 200 chars each, up to 3}
    \"\"\"

metadata:
    id:            "P02_summary"
    type:          "summary"
    programme_id:  "P02"
    programme_name: "MA International Accounting"
    source:        "p02.md"

Purpose: recommendation queries ("recommend suitable master's programmes")
rank whole-programme summaries instead of section fragments.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

ROOT_DIR = Path(__file__).resolve().parent.parent
PROGRAMMES_JSON = ROOT_DIR / "data" / "programmes.json"

COURSES_SECTIONS = (
    "Course Description",
    "Programme Content",
    "Programme Structure & Courses",
)
HIGHLIGHT_SECTIONS = (
    "Professional Accreditation",
    "Professional Recognition",
    "Bonus Features",
    "Programme Features",
    "Career",
    "Career Prospects",
    "Hong Kong Future Talents Scholarship Scheme for Advanced Studies",
    "Did You Know?",
)

COURSES_CHARS = 500
HIGHLIGHT_CHARS = 200
MAX_HIGHLIGHTS = 3


def _section_content(programme: dict, title: str) -> str:
    for section in programme["sections"]:
        if section["title"] == title:
            return section["content"]
    return ""


def _build_summary(programme: dict) -> str:
    name = programme.get("name") or programme["programme_id"]
    aims = _section_content(programme, "Programme Aims and Objectives")

    courses = ""
    for title in COURSES_SECTIONS:
        content = _section_content(programme, title)
        if content:
            courses = content
            break

    highlights = []
    for title in HIGHLIGHT_SECTIONS:
        content = _section_content(programme, title)
        if content and len(highlights) < MAX_HIGHLIGHTS:
            highlights.append(f"{title}:\n{content[:HIGHLIGHT_CHARS]}")

    return f"""Programme:
{name}

Overview:
{aims}

Courses:
{courses[:COURSES_CHARS]}

Highlights:
{chr(10).join(highlights) if highlights else "-"}""".strip()


def build_summaries(programmes: list | None = None) -> list[Document]:
    """One summary Document per programme."""
    if programmes is None:
        programmes = json.loads(PROGRAMMES_JSON.read_text(encoding="utf-8"))

    documents: list[Document] = []
    for programme in programmes:
        programme_id = programme["programme_id"]
        name = programme.get("name") or programme_id
        documents.append(
            Document(
                page_content=_build_summary(programme),
                metadata={
                    "id": f"{programme_id}_summary",
                    "type": "summary",
                    "programme_id": programme_id,
                    "programme_name": name,
                    "source": Path(programme["source_file"]).name,
                },
            )
        )
    return documents


def build_all() -> tuple[list[Document], list[str]]:
    docs = build_summaries()
    return docs, [d.metadata["id"] for d in docs]


def main() -> None:
    docs, ids = build_all()
    print(f"built {len(docs)} summary documents")
    sample = docs[0]
    print(f"\n--- {sample.metadata['id']} ---")
    print(sample.page_content[:600])


if __name__ == "__main__":
    main()

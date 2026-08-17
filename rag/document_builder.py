"""Build child Documents from parsed Programme objects.

Input : ``data/programmes.json`` (output of ``rag/parser.py``)
Output: ``List[Document]`` -- one Document per programme section, ready for
        Chroma embedding (``chroma.add_documents(documents=docs, ids=ids)``).

Each section becomes a child document carrying the programme context:

    page_content:
        Programme:
        MSc Mechanical Engineering

        Section:
        Entrance Requirements

        Applicants must satisfy ...

    metadata:
        id:           "P66_Entrance Requirements"
        programme_id: "P66"
        programme_name: "MSc Mechanical Engineering"
        section:      "Entrance Requirements"
        category:     "B" | "C"
        source:       "p66.md"
        optional:     true   (only for category C sections)

Category semantics (docs/programme_schema.md §4/§5):
  * B -- core retrieval knowledge: embedded normally.
  * C -- optional info (scholarship, accreditation, career, ...): embedded but
         tagged ``optional: true`` so retrieval can boost/suppress by intent.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

ROOT_DIR = Path(__file__).resolve().parent.parent
PROGRAMMES_JSON = ROOT_DIR / "data" / "programmes.json"

CATEGORY_B = "B"
CATEGORY_C = "C"


def _page_content(name: str, section_title: str, content: str) -> str:
    return (
        f"Programme:\n{name}\n\n"
        f"Section:\n{section_title}\n\n"
        f"{content}"
    )


def build_documents(programmes: list | None = None) -> list[Document]:
    """Convert parsed programmes into one child Document per section.

    ``programmes``: list of Programme objects (from ``rag/parser.py``).
    Defaults to loading ``data/programmes.json``.
    """
    if programmes is None:
        programmes = json.loads(PROGRAMMES_JSON.read_text(encoding="utf-8"))

    documents: list[Document] = []
    id_counts: dict[str, int] = {}
    for prog in programmes:
        programme_id = prog["programme_id"]
        name = prog.get("name") or programme_id
        source = Path(prog["source_file"]).name  # e.g. "p66.md"

        for section in prog["sections"]:
            title = section["title"]
            category = section["category"]

            # disambiguate repeated sections (e.g. p18 has two "Course
            # Description" blocks) so Chroma ids stay unique
            base_id = f"{programme_id}_{title}"
            id_counts[base_id] = id_counts.get(base_id, 0) + 1
            doc_id = (
                base_id
                if id_counts[base_id] == 1
                else f"{base_id}_{id_counts[base_id]}"
            )

            metadata = {
                "id": doc_id,
                "programme_id": programme_id,
                "programme_name": name,
                "section": title,
                "category": category,
                "source": source,
            }
            if category == CATEGORY_C:
                metadata["optional"] = True

            documents.append(
                Document(
                    page_content=_page_content(name, title, section["content"]),
                    metadata=metadata,
                )
            )

    return documents


def get_ids(documents: list[Document]) -> list[str]:
    """Chroma document ids (used as the ``ids`` argument of add_documents)."""
    return [d.metadata["id"] for d in documents]


def build_all() -> tuple[list[Document], list[str]]:
    """Convenience: load programmes.json, build documents and ids."""
    docs = build_documents()
    return docs, get_ids(docs)


def main() -> None:
    docs, ids = build_all()
    print(f"built {len(docs)} child documents")

    from collections import Counter

    cats = Counter(d.metadata["category"] for d in docs)
    print(f"categories: {dict(cats)}")
    print(f"optional-tagged: {sum('optional' in d.metadata for d in docs)}")

    sample = docs[0]
    print("\n--- sample ---")
    print(f"id:      {sample.metadata['id']}")
    print(f"metadata: {sample.metadata}")
    print(f"page_content:\n{sample.page_content[:300]}")


if __name__ == "__main__":
    main()

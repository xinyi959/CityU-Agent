"""Build metadata documents for the exact-fact query index (v1).

Converts Category A structured metadata of each Programme object
(``data/programmes.json``) into one natural-language Document per programme,
for the dedicated Chroma collection ``programme_metadata``.

Metadata documents serve exact factual queries (tuition fee, deadline,
study period, mode of study, ...). They do NOT participate in programme
recommendation.

page_content (formatted like the source page):

    Programme:
    MSc Mechanical Engineering

    Year of Entry:
    2026

    Mode of Study:
    Combined

    Tuition Fee:
    Local Students: HK$8,100 per credit
    Non-local Students: HK$8,100 per credit

metadata:
    id:            "P66_metadata"
    programme_id:  "P66"
    programme_name: "MSc Mechanical Engineering"
    type:          "metadata"
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

ROOT_DIR = Path(__file__).resolve().parent.parent
PROGRAMMES_JSON = ROOT_DIR / "data" / "programmes.json"

# snake key -> human-readable label (parser output order)
FIELD_LABELS = {
    "year_of_entry": "Year of Entry",
    "application_deadline": "Application Deadline",
    "mode_of_study": "Mode of Study",
    "mode_of_funding": "Mode of Funding",
    "indicative_intake_target": "Indicative Intake Target",
    "minimum_no_of_credits_required": "Minimum No. of Credits Required",
    "class_schedule": "Class Schedule",
    "normal_study_period": "Normal Study Period",
    "maximum_study_period": "Maximum Study Period",
    "mode_of_processing": "Mode of Processing",
    "tuition_fee": "Tuition Fee",
    "programme_website": "Programme Website",
    "intermediate_award": "Intermediate Award",
    "deadline":"Application Deadline"
}


def value_to_text(value):
    """Render a metadata value as lines of natural language."""
    if value is None:
        return None
    if isinstance(value, dict):
        if "local" in value or "non_local" in value or "source" in value:
            # tuition fee style: Local / Non-local / Source
            parts = []
            for label, key in (
                ("Local Students", "local"),
                ("Non-local Students", "non_local"),
                ("Source", "source"),
            ):
                if value.get(key):
                    parts.append(f"{label}: {value[key]}")
            return "\n".join(parts) or None
        if "raw" in value or "iso" in value:
            # deadline style: prefer human-readable dates
            dates = value.get("raw") or value.get("iso") or []
            return "\n".join(dates) or None
        return "\n".join(f"{k}: {v}" for k, v in value.items() if v) or None
    if isinstance(value, list):
        return "\n".join(str(v) for v in value) or None
    return str(value)


def build_metadata_document(programme: dict) -> Document:
    """One metadata Document for a single Programme object."""
    programme_id = programme["programme_id"]
    name = programme.get("name") or programme_id
    metadata = programme.get("metadata", {})

    blocks = [f"Programme:\n{name}"]
    for key, label in FIELD_LABELS.items():
        text = value_to_text(metadata.get(key))
        if text:
            blocks.append(f"{label}:\n{text}")

    return Document(
        page_content="\n\n".join(blocks),
        metadata={
            "id": f"{programme_id}_metadata",
            "programme_id": programme_id,
            "programme_name": name,
            "type": "metadata",
        },
    )


def build_metadata_documents(programmes: list | None = None) -> list[Document]:
    """One metadata Document per programme (64 docs)."""
    if programmes is None:
        programmes = json.loads(PROGRAMMES_JSON.read_text(encoding="utf-8"))
    return [build_metadata_document(p) for p in programmes]


def build_field_document(programme: dict, field: str) -> Document:
    """Direct-lookup Document for a single metadata field of a programme.

    Used by the metadata retriever when the query resolves to a specific
    programme + field (e.g. tuition fee of P53). For tuition fee, both
    local and non-local rates are included by default.
    """
    programme_id = programme["programme_id"]
    name = programme.get("name") or programme_id
    metadata = programme.get("metadata", {})

    if field not in FIELD_LABELS:
        return build_metadata_document(programme)

    label = FIELD_LABELS[field]
    text = value_to_text(metadata.get(field))
    page_content = f"Programme:\n{name}"
    if text:
        page_content += f"\n\n{label}:\n{text}"

    return Document(
        page_content=page_content,
        metadata={
            "id": f"{programme_id}_metadata_{field}",
            "programme_id": programme_id,
            "programme_name": name,
            "type": "metadata",
            "field": field,
        },
    )


def build_all() -> tuple[list[Document], list[str]]:
    docs = build_metadata_documents()
    return docs, [d.metadata["id"] for d in docs]


def main() -> None:
    docs, ids = build_all()
    print(f"built {len(docs)} metadata documents")
    sample = next(d for d in docs if d.metadata["programme_id"] == "P66")
    print(f"\n--- {sample.metadata['id']} ---")
    print(sample.page_content)


if __name__ == "__main__":
    main()

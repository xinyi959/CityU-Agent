"""Metadata retriever: structured resolution + vector fallback (Evidence).

Exact facts (tuition fee, deadline, duration, ...) are NOT semantic
knowledge, so when the query identifies a programme we resolve it directly
against ``data/programmes.json`` and return the exact structured value.

Resolved -> single Evidence with score 1.0 (exact match). The source URL is
kept OUT of ``content`` and stored in ``evidence.metadata["url"]`` (the LLM
gets content only; the citation formatter reads metadata).
Fallback -> Evidence objects from ``programme_metadata`` vector search.
"""

from rag.evidence import Evidence
from rag.metadata_builder import FIELD_LABELS, build_metadata_document, value_to_text
from rag.programme_resolver import (
    extract_field,
    resolve_programme_scope,
)
from rag.retriever import retrieve_metadata


def _render_fee(value: dict) -> str:
    """Local / Non-local on separate label/value lines (no Source line)."""
    groups = []
    for label, key in (
        ("Local Students", "local"),
        ("Non-local Students", "non_local"),
    ):
        if value.get(key):
            groups.append(f"{label}:\n{value[key]}")
    if not groups:
        # Programme has a fee source URL but no parsed local/non-local rates
        # (e.g. P83). Never hand the generator an empty fee block.
        return "Not available in our records. See the programme website."
    return "\n\n".join(groups)


def _split_source(value):
    """Return (clean_value, url): pull 'source' out of a metadata dict."""
    if isinstance(value, dict) and "source" in value:
        rest = {k: v for k, v in value.items() if k != "source"}
        return rest, value.get("source")
    return value, None


def _evidence_for_programme(programme: dict, field: str) -> Evidence:
    """Exact structured lookup for ONE resolved programme + field.

    Shared by the single-programme path and the recommendation-scoped path
    (which loops over several programme ids).
    """
    pid = programme["programme_id"]
    raw = programme.get("metadata", {}).get(field)
    section = FIELD_LABELS.get(field, field or "metadata")

    url = None
    if field == "tuition_fee" and isinstance(raw, dict):
        content = _render_fee(raw)
        url = raw.get("source")
    elif raw is not None:
        clean, url = _split_source(raw)
        content = value_to_text(clean) or ""
    else:
        content = build_metadata_document(programme).page_content

    return Evidence(
        id=f"{pid}-{field or 'metadata'}",
        programme_id=pid,
        section=section,
        content=content,
        score=1.0,
        source_type="metadata",
        metadata={"url": url} if url else None,
    )


def metadata_retriever_node(state):
    query = state["query"]

    field = (
        state.get("field")
        or extract_field(query)
    )

    # Resolve the programme(s) to scope this sub-question to: explicit query
    # mention -> scope set (this turn's recommendation / previous turn's
    # persisted set) -> router inferred ref -> message history.
    programmes = resolve_programme_scope(
        query,
        programme_ref=state.get("programme_ref"),
        messages=state.get("messages", []),
        scope_ids=state.get("programme_ids") or [],
    )

    if programmes:
        return {
            "evidence": [
                _evidence_for_programme(programme, field)
                for programme in programmes
            ],
            "resolved_programme_refs": [
                {
                    "programme_id": programme["programme_id"],
                    "programme_name": programme.get("name"),
                }
                for programme in programmes
            ],
        }

    # fallback: no resolvable programme -> semantic search on metadata index
    evidence = [
        Evidence(
            id=f"{doc.metadata['programme_id']}-metadata-{i}",
            programme_id=doc.metadata["programme_id"],
            section=(
                FIELD_LABELS.get(field, field)
                if field
                else "metadata"
            ),
            content=doc.page_content,
            score=score,
            source_type="metadata",
        )
        for i, (doc, score) in enumerate(retrieve_metadata(query))
    ]
    return {
        "evidence": evidence,
    }

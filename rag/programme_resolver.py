"""Programme reference extraction and resolution.

Given a user query, identify:

1. which programme is being asked about:

       "What is the tuition fee of MSc Computer Science?"  -> P53 (by name)
       "P53 tuition fee"                                   -> P53 (by code)

2. which structured metadata field is requested:

       "tuition fee"   -> tuition_fee
       "deadline"      -> application_deadline
       "credits"       -> minimum_no_of_credits_required

Exact facts (fees, deadlines, durations) are resolved against the structured
``data/programmes.json`` instead of being answered via semantic search.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PROGRAMMES_JSON = ROOT_DIR / "data" / "programmes.json"

_ID_RE = re.compile(r"\bP\d{2,3}\b", re.IGNORECASE)

# field -> keywords (more specific phrases first)
FIELD_KEYWORDS = [
    ("mode_of_study", ["mode of study", "study mode", "full-time", "part-time"]),
    (
        "mode_of_funding",
        ["mode of funding", "funding", "government-funded", "subsidised", "subsidized"],
    ),
    ("application_deadline", ["deadline", "closing date", "apply by"]),
    ("tuition_fee", ["tuition", "fee", "fees", "cost"]),
    (
        "normal_study_period",
        ["duration", "study period", "how long", "normal study"],
    ),
    (
        "minimum_no_of_credits_required",
        ["credit units", "credit", "credits"],
    ),
    ("indicative_intake_target", ["intake", "quota", "places"]),
    ("class_schedule", ["class schedule", "schedule", "classes"]),
    ("programme_website", ["website", "homepage", "link"]),
    ("year_of_entry", ["year of entry", "entry year"]),
]


@lru_cache(maxsize=1)
def _load_programmes() -> list:
    return json.loads(PROGRAMMES_JSON.read_text(encoding="utf-8"))


def get_programmes() -> list:
    return _load_programmes()


def extract_programme_ref(query: str) -> dict:
    """Return {"programme_id": ..., "programme_name": ...} found in the query.

    Programme code (P53) is tried first, then longest programme-name match.
    """
    q = query.lower()

    m = _ID_RE.search(q)
    if m:
        return {"programme_id": m.group(0).upper(), "programme_name": None}

    name = _match_programme_name(q)
    if name:
        return {"programme_id": None, "programme_name": name}

    return {"programme_id": None, "programme_name": None}


def _match_programme_name(q: str) -> str | None:
    """Longest programme name that appears verbatim in the query."""
    best = None
    for p in _load_programmes():
        original = p.get("name")
        if not original:
            continue
        lowered = original.lower()
        if lowered in q and (best is None or len(lowered) > len(best[1])):
            best = (original, lowered)
    return best[0] if best else None


def find_programme(ref: dict | None) -> dict | None:
    """Resolve a programme reference to the full Programme object (or None)."""
    if not ref or (not ref["programme_id"] and not ref["programme_name"]):
        return None
    for p in _load_programmes():
        if ref["programme_id"] and p["programme_id"] == ref["programme_id"]:
            return p
        if ref["programme_name"] and p.get("name") == ref["programme_name"]:
            return p
    return None


def resolve_programme(query: str) -> dict | None:
    """Convenience: extract ref from a query and resolve to a Programme."""
    return find_programme(extract_programme_ref(query))


def _message_text(msg) -> str | None:
    """Extract plain text from a message (LangChain BaseMessage or dict).

    Mirrors ``agent/graph.py::input_adapter``: content may be a str or a list
    of content blocks; only text blocks are concatenated.
    """
    if hasattr(msg, "content"):
        content = msg.content
    elif isinstance(msg, dict):
        content = msg.get("content")
    else:
        return None

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return None


def _human_message_texts(messages: list) -> list[str]:
    """Recent-first plain texts of human (or unknown-format) messages."""
    texts = []
    for m in reversed(messages):
        if hasattr(m, "type"):
            msg_type = m.type
        elif isinstance(m, dict):
            msg_type = m.get("type")
        else:
            msg_type = None
        if msg_type is not None and msg_type != "human":
            continue
        text = _message_text(m)
        if text:
            texts.append(text)
    return texts


def resolve_programme_scope(
    query: str,
    programme_ref: dict | None = None,
    messages: list | None = None,
    scope_ids: list | None = None,
) -> list[dict]:
    """Resolve the programme(s) a query refers to, as a LIST (retrieval scope).

    Candidates are tried in priority order:

    1. explicit programme in the CURRENT query text (P-code / name) --
       strongest: the user named a programme THIS turn (topic switch).
    2. ``scope_ids`` -- the recommendation set resolved this turn, or the
       set persisted by the PREVIOUS turn (dispatcher passes both as
       ``programme_ids``). Beats the router's inferred single ref so
       "Any apply requirement I should fulfill?" after a recommendation
       stays scoped to the whole recommended set rather than the one
       programme the router happens to re-emit.
    3. router ``programme_ref`` (single, inferred from history when the
       query omits a referent).
    4. text rules over recent human messages (most recent first).

    Returns [] when nothing resolves -- the caller falls back to a
    whole-corpus semantic search.
    """
    # 1. explicit this-turn mention
    if query:
        programme = find_programme(extract_programme_ref(query))
        if programme:
            return [programme]

    # 2. scope set (this turn's recommendation or the previous turn's set)
    programmes = []
    for pid in scope_ids or []:
        programme = find_programme(
            {"programme_id": pid, "programme_name": None}
        )
        if programme and programme not in programmes:
            programmes.append(programme)
    if programmes:
        return programmes

    # 3. router ref (inferred single)
    if programme_ref and (
        programme_ref.get("programme_id") or programme_ref.get("programme_name")
    ):
        programme = find_programme(programme_ref)
        if programme:
            return [programme]

    # 4. recent human messages
    if messages:
        for text in _human_message_texts(messages):
            programme = find_programme(extract_programme_ref(text))
            if programme:
                return [programme]

    return []


def resolve_programme_ref(
    query: str,
    programme_ref: dict | None = None,
    messages: list | None = None,
    scope_ids: list | None = None,
) -> dict | None:
    """Single-programme convenience wrapper over :func:`resolve_programme_scope`.

    Returns the programme only when the scope resolves to exactly ONE; a
    multi-programme scope is not a single referent, so it returns None.
    """
    programmes = resolve_programme_scope(
        query,
        programme_ref=programme_ref,
        messages=messages,
        scope_ids=scope_ids,
    )
    return programmes[0] if len(programmes) == 1 else None


def extract_field(query: str) -> str | None:
    """Return the metadata field key the query asks about (or None)."""
    q = query.lower()
    for field, keywords in FIELD_KEYWORDS:
        for kw in keywords:
            if _match(q, kw):
                return field
    return None


def _match(q: str, keyword: str) -> bool:
    if keyword == "credit" or keyword == "credits":
        # \bcredit\w* matches credit/credits but not accreditation/accredited
        return bool(re.search(r"\bcredit\w*", q))
    return keyword in q


if __name__ == "__main__":
    for ex in [
        "What is the tuition fee of MSc Computer Science?",
        "P53 tuition fee",
        "When is the application deadline of MA International Accounting?",
        "How many credits does P66 require?",
        "Is the programme accredited?",
    ]:
        ref = extract_programme_ref(ex)
        prog = find_programme(ref)
        field = extract_field(ex)
        print(
            f"{ex[:55]:<57} -> ref={ref} field={field} "
            f"resolved={prog['programme_id'] if prog else None}"
        )

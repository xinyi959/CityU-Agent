"""Router: one decision per sub-question of a (possibly compound) query.

Phase 0 probe findings this node encodes:

1. Split quality is good, but the nested-list structured output of
   deepseek-v4-flash is unstable at temperature=0: `field` sometimes
   decodes as None while the free-text `sub_query` absorbs the leaked
   JSON ("field: tuition_fee, programme: ..."). So we do NOT trust the
   schema alone -- a deterministic repair layer fills missing fields
   from the sub_query wording, and a blind retry is the backstop.

   The model outputs ONE routing signal per sub-question (`field`);
   `retrieval_type` is derived from it (retrieval_type_of) so a
   mismatched (field, retrieval_type) pair -- the old two-axis
   coordination failure -- is structurally impossible.
2. The model tends to re-emit already-answered history questions as
   extra decisions in multi-turn compound turns -- the prompt forbids it.
3. Programme codes ("P53") are not mapped to names by the model -- the
   downstream resolve_programme_ref() handles codes via text rules, so a
   missing programme_ref is acceptable (not a router failure).

Fallback: if both attempts come back invalid, a rule-based decision is
built from the full query (reusing the v1 keyword router in rag/router.py
and the metadata field extractor in rag/programme_resolver.py) so the
graph always receives a usable plan.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import model
from agent.state.router_schema import (
    ProgrammeRefModel,
    RouterDecisionList,
    RouterSubDecision,
    retrieval_type_of,
)
from rag.programme_resolver import extract_programme_ref

# 输出intent, retrieval type, field
ROUTER_PROMPT = """
You are the routing agent for CityUHK postgraduate assistant.

The user query may contain MULTIPLE sub-questions (e.g. "What are the
English requirements and tuition fee of MSc Computer Science?").

Rules:

1. Only consider the LATEST user message. Never re-emit questions that
   were already asked and answered in earlier turns.
2. Split the latest message into ONE sub-decision per sub-question,
   keeping the original order.
3. A simple single-question message produces exactly one decision.
4. At most 4 decisions; keep the first 4 if there are more.
5. `sub_query` must be the sub-question wording only -- no field names,
   no programme codes, no extra annotations.

## Intent (top-level, one value for the whole turn)

qa:
- asking factual programme information
- asking about requirements, fees, duration, details

recommendation:
- asking which programme suits them
- asking for suggestions

comparison:
- comparing multiple programmes

## field (per sub-decision)

One routing signal per sub-question: pick the SINGLE `field` that the
sub-question asks about. The retrieval path follows automatically from
the field -- do NOT output a retrieval type.

metadata fields (exact structured facts):
- tuition_fee   -- tuition fee, fees, cost, how much
- deadline      -- application deadline, closing date, apply by
- duration      -- how long, study period, length of
- credit        -- credit units, credits
- study_mode    -- mode of study, full-time, part-time

section fields (detailed programme information):
- entrance_requirement -- entrance/admission requirements, English
                          requirements, IELTS/TOEFL
- curriculum           -- curriculum, courses, syllabus

summary (programme recommendation, overview, comparison):
- "summary"

Common mappings: "English requirements" -> entrance_requirement,
"curriculum"/"courses" -> curriculum, "how long" -> duration.

## programme_ref

- Top-level `programme_ref`: the programme shared by the whole question
  (inherited from the conversation when the query omits it).
- Per-sub-decision `programme_ref`: set ONLY when that sub-question
  refers to a DIFFERENT programme than the top-level one (cross-programme
  compound questions). Otherwise leave it empty.

Return structured JSON only.
"""

router_llm = model.with_structured_output(RouterDecisionList)

MAX_DECISIONS = 4

# ---------------------------------------------------------------------
# Deterministic field repair (no LLM): sub_query wording -> router field
# literal. Order matters -- more specific phrases first.
# ---------------------------------------------------------------------

FIELD_KEYWORDS = [
    ("tuition_fee", ["tuition", "fees", "fee", "cost", "how much"]),
    (
        "deadline",
        [
            "application deadline",
            "closing date",
            "application date",
            "deadline",
            "apply by",
        ],
    ),
    ("duration", ["study period", "how long", "duration", "length of"]),
    ("credit", ["credit units", "credits", "credit"]),
    (
        "study_mode",
        [
            "mode of study",
            "study mode",
            "full-time",
            "part-time",
            "full time",
            "part time",
        ],
    ),
    (
        "entrance_requirement",
        [
            "entrance requirement",
            "admission requirement",
            "english requirement",
            "english language",
            "english",
            "ielts",
            "toefl",
            "requirement",
        ],
    ),
    ("curriculum", ["curriculum", "courses", "course", "syllabus"]),
]

# rag/programme_resolver.extract_field() key -> router field literal
METADATA_FIELD_MAP = {
    "tuition_fee": "tuition_fee",
    "application_deadline": "deadline",
    "normal_study_period": "duration",
    "minimum_no_of_credits_required": "credit",
    "mode_of_study": "study_mode",
}


def _match(q: str, keyword: str) -> bool:
    """Word-boundary match: "fee" hits fee/fees but not feedback/coffee."""
    if " " in keyword:
        return keyword in q
    return bool(re.search(rf"\b{re.escape(keyword)}\w*", q))


def _repair_field(decision: RouterSubDecision) -> RouterSubDecision:
    """Fill a missing `field` from the sub_query wording.

    No-op when the model already emitted a field ("summary" included).
    """
    if decision.field is not None:
        return decision
    q = decision.sub_query.lower()
    for field, keywords in FIELD_KEYWORDS:
        if any(_match(q, kw) for kw in keywords):
            decision.field = field
            break
    return decision


def _is_valid(decision: RouterSubDecision) -> bool:
    """A decision is usable if it can drive one retriever call.

    retrieval_type is derived from field, so the only failures left are
    a missing/leaked sub_query and an unfilled field.
    """
    if not decision.sub_query or len(decision.sub_query) < 3:
        return False
    # leaked structured-output JSON inside the free-text sub_query
    if "field:" in decision.sub_query or "programme:" in decision.sub_query:
        return False

    return decision.field is not None  # repair could not fill it


def _repair_programme_ref(decision: RouterSubDecision) -> RouterSubDecision:
    """Fill an empty per-decision programme_ref from the sub_query text.

    The model sometimes drops programme_ref entirely (Phase-0 decode
    instability). The sub_query almost always carries the programme name
    (the model expands omitted referents), so we re-extract it
    deterministically. No-op when the sub_query has no programme mention.
    """
    if decision.programme_ref and (
        decision.programme_ref.programme_id
        or decision.programme_ref.programme_name
    ):
        return decision
    ref = extract_programme_ref(decision.sub_query)
    if ref.get("programme_id") or ref.get("programme_name"):
        decision.programme_ref = ProgrammeRefModel(**ref)
    return decision


def _repair_top_programme_ref(
    decision_list: RouterDecisionList,
) -> RouterDecisionList:
    """Fill an empty top-level programme_ref from the first decision's ref.

    The top-level ref is the programme shared by the whole question; when
    the model drops it, the first sub-question's referent is a reasonable
    default (cross-programme compounds carry their own per-decision refs).
    """
    if decision_list.programme_ref and (
        decision_list.programme_ref.programme_id
        or decision_list.programme_ref.programme_name
    ):
        return decision_list
    for d in decision_list.decisions:
        if d.programme_ref and (
            d.programme_ref.programme_id or d.programme_ref.programme_name
        ):
            decision_list.programme_ref = d.programme_ref
            break
    return decision_list


def _prepare(decision_list: RouterDecisionList) -> RouterDecisionList | None:
    """Cap decision count, repair missing fields and programme refs.

    Defensive: a structured-output decode can yield None or an empty
    ``decisions`` list without raising; return it untouched so the caller's
    validity check rejects it and falls through to the next attempt.
    """
    if decision_list is None or not decision_list.decisions:
        return decision_list
    decision_list.decisions = decision_list.decisions[:MAX_DECISIONS]
    for d in decision_list.decisions:
        _repair_field(d)
        _repair_programme_ref(d)
    _repair_top_programme_ref(decision_list)
    return decision_list


def _to_programme_ref(ref) -> dict | None:
    if not ref:
        return None
    return {
        "programme_id": ref.programme_id,
        "programme_name": ref.programme_name,
    }


def _fallback_decision_list(query: str) -> RouterDecisionList:
    """Rule-based plan when the LLM router fails twice.

    Reuses the v1 keyword router (rag/router.py) and the metadata field
    extractor (rag/programme_resolver.py) as a deterministic safety net.
    """
    from rag.programme_resolver import extract_field, extract_programme_ref
    from rag.router import classify_query

    retrieval_type = classify_query(query) if query else "section"
    q = (query or "").lower()

    field = None
    if retrieval_type == "metadata":
        field = METADATA_FIELD_MAP.get(extract_field(query) or "")
        if field is None:
            # metadata keyword hit but no field matched -> detailed
            # requirements instead of a bare metadata dump
            field = "entrance_requirement"
    elif retrieval_type == "summary":
        field = "summary"
    else:
        field = (
            "curriculum"
            if re.search(r"curriculum|courses?\b", q)
            else "entrance_requirement"
        )

    programme_ref = extract_programme_ref(query)
    return RouterDecisionList(
        intent="qa",
        programme_ref=(
            ProgrammeRefModel(**programme_ref) if programme_ref else None
        ),
        decisions=[
            RouterSubDecision(
                field=field,
                sub_query=query,
            )
        ],
    )


def router_node(state):
    messages = state.get("messages") or []
    query = state.get("query") or ""

    # Query-only input (CLI / direct invoke): synthesize the user turn from
    # `query` so the LLM router has a message to route on.
    if not messages and query:
        messages = [HumanMessage(content=query)]

    decision_list = None
    for _ in range(2):
        try:
            candidate = router_llm.invoke(
                [SystemMessage(content=ROUTER_PROMPT), *messages]
            )
        except Exception as exc:  # structured-output parse failure
            print("ROUTER RETRY (parse error):", type(exc).__name__)
            continue

        # structured output can decode to None WITHOUT raising (empty /
        # refusal reply from the model). _prepare would crash on .decisions,
        # so reject it here and retry -- the rule fallback is the backstop.
        if candidate is None or not candidate.decisions:
            print("ROUTER RETRY (empty structured output):", type(candidate).__name__)
            continue

        candidate = _prepare(candidate)
        if candidate.decisions and all(_is_valid(d) for d in candidate.decisions):
            decision_list = candidate
            break
        print(
            "ROUTER RETRY (invalid plan):",
            [
                (retrieval_type_of(d.field), d.field)
                for d in candidate.decisions
            ],
        )

    if decision_list is None:
        decision_list = _fallback_decision_list(query)
        print("ROUTER FALLBACK (rule-based)")

    print(
        "ROUTER PLAN:",
        decision_list.intent,
        [
            (retrieval_type_of(d.field), d.field)
            for d in decision_list.decisions
        ],
    )

    return {
        "intent": decision_list.intent,
        "programme_ref": _to_programme_ref(decision_list.programme_ref),
        "decisions": [
            {
                **d.model_dump(),
                # derived routing path (kept on the dict for the
                # dispatcher and for backward-compatible consumers)
                "retrieval_type": retrieval_type_of(d.field),
            }
            for d in decision_list.decisions
        ],
    }

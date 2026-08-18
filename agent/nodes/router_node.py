"""Router: one decision per sub-question of a (possibly compound) query.

Phase 0 probe findings this node encodes:

1. Split quality is good, but the nested-list structured output of
   deepseek-v4-flash is unstable at temperature=0: `field` sometimes
   decodes as None while the free-text `sub_query` absorbs the leaked
   JSON ("field: tuition_fee, programme: ..."). So we do NOT trust the
   schema alone -- a deterministic repair layer fills missing fields
   from the sub_query wording, and a blind retry is the backstop.
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

from langchain_core.messages import SystemMessage

from agent.llm import model
from agent.state.router_schema import (
    ProgrammeRefModel,
    RouterDecisionList,
    RouterSubDecision,
)

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

## Retrieval Type (per sub-decision)

metadata:
- exact structured facts
- tuition fee
- deadline
- duration
- credit
- study mode

section:
- detailed programme information
- entrance requirements
- curriculum
- course information

summary:
- programme recommendation
- programme overview
- programme comparison

## field (per sub-decision)

metadata fields: tuition_fee, deadline, duration, credit, study_mode
section fields:  entrance_requirement, curriculum
summary fields:  leave `field` empty.

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

METADATA_FIELDS = {"tuition_fee", "deadline", "duration", "credit", "study_mode"}
SECTION_FIELDS = {"entrance_requirement", "curriculum"}

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

    Only applies to metadata/section decisions; summary decisions are
    legitimately field-less.
    """
    if decision.field is not None or decision.retrieval_type == "summary":
        return decision
    q = decision.sub_query.lower()
    for field, keywords in FIELD_KEYWORDS:
        if any(_match(q, kw) for kw in keywords):
            decision.field = field
            break
    return decision


def _is_valid(decision: RouterSubDecision) -> bool:
    """A decision is usable if it can drive one retriever call."""
    if not decision.sub_query or len(decision.sub_query) < 3:
        return False
    # leaked structured-output JSON inside the free-text sub_query
    if "field:" in decision.sub_query or "programme:" in decision.sub_query:
        return False

    if decision.retrieval_type == "summary":
        return decision.field is None

    if decision.field is None:
        return False  # repair could not fill it

    if decision.field in METADATA_FIELDS and decision.retrieval_type != "metadata":
        return False
    if decision.field in SECTION_FIELDS and decision.retrieval_type != "section":
        return False
    return True


def _prepare(decision_list: RouterDecisionList) -> RouterDecisionList:
    """Cap decision count and repair missing fields."""
    decision_list.decisions = decision_list.decisions[:MAX_DECISIONS]
    for d in decision_list.decisions:
        _repair_field(d)
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

    field = None
    if retrieval_type == "metadata":
        field = METADATA_FIELD_MAP.get(extract_field(query) or "")
        if field is None:
            retrieval_type = "section"  # keyword hit but no field matched
    elif retrieval_type == "section":
        q = query.lower()
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
                retrieval_type=retrieval_type,
                field=field,
                sub_query=query,
            )
        ],
    )


def router_node(state):
    messages = state["messages"]
    query = state.get("query") or ""

    decision_list = None
    for _ in range(2):
        try:
            candidate = router_llm.invoke(
                [SystemMessage(content=ROUTER_PROMPT), *messages]
            )
        except Exception as exc:  # structured-output parse failure
            print("ROUTER RETRY (parse error):", type(exc).__name__)
            continue

        candidate = _prepare(candidate)
        if candidate.decisions and all(_is_valid(d) for d in candidate.decisions):
            decision_list = candidate
            break
        print(
            "ROUTER RETRY (invalid plan):",
            [(d.retrieval_type, d.field) for d in candidate.decisions],
        )

    if decision_list is None:
        decision_list = _fallback_decision_list(query)
        print("ROUTER FALLBACK (rule-based)")

    print(
        "ROUTER PLAN:",
        decision_list.intent,
        [(d.retrieval_type, d.field) for d in decision_list.decisions],
    )

    first = decision_list.decisions[0] if decision_list.decisions else None
    return {
        "intent": decision_list.intent,
        # Back-compat with the single-decision graph: routing signals of the
        # first decision. Phase 2 (dispatcher) consumes `decisions` instead.
        "retrieval_type": first.retrieval_type if first else "section",
        "field": first.field if first else None,
        "programme_ref": _to_programme_ref(decision_list.programme_ref),
        "decisions": [d.model_dump() for d in decision_list.decisions],
    }

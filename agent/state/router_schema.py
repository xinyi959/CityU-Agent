from typing import Literal
from pydantic import BaseModel

# Single routing axis: `field`. The retrieval path is DERIVED from the
# field (retrieval_type_of), never output by the model. The old schema
# asked the model to coordinate (field, retrieval_type) pairs; mismatched
# pairs and field=None leaks were the main structured-output decode
# failure in the Phase 0 probe.

METADATA_FIELDS = frozenset(
    {"tuition_fee", "deadline", "duration", "credit", "study_mode"}
)
SECTION_FIELDS = frozenset({"entrance_requirement", "curriculum"})

# field literal values the model may emit. "summary" is a first-class
# value: recommendation / overview / comparison sub-questions.
RouterField = Literal[
    "tuition_fee",
    "deadline",
    "duration",
    "credit",
    "study_mode",
    "entrance_requirement",
    "curriculum",
    "summary",
]


def retrieval_type_of(field: str | None) -> str:
    """Derive the retrieval path from the router field (pure function).

    field -> retrieval_type is 1:1 in the valid decision space:

      5 metadata fields -> "metadata"
      2 section fields  -> "section"
      "summary"         -> "summary"

    ``None`` (a field the deterministic repair could not fill) is invalid
    and never reaches the dispatcher.
    """
    if field in METADATA_FIELDS:
        return "metadata"
    if field in SECTION_FIELDS:
        return "section"
    return "summary"


class ProgrammeRefModel(BaseModel):
    programme_id: str | None = None
    programme_name: str | None = None


class RouterSubDecision(BaseModel):
    """One sub-question of a (possibly compound) user query.

    Single routing signal: ``field``. ``retrieval_type`` is NOT part of
    the model output -- the dispatcher derives it via
    ``retrieval_type_of(field)``, so a mismatched (field, retrieval_type)
    pair is structurally impossible.

    intent lives at the top level (RouterDecisionList) -- a compound turn
    is almost always one intent -- so it is intentionally NOT repeated
    here: per-decision intent was the main source of structured-output
    decode instability in the Phase 0 probe.
    """

    field: RouterField | None = None

    sub_query: str

    # Set ONLY when this sub-question refers to a DIFFERENT programme than
    # the top-level programme_ref (cross-programme compound questions).
    # Otherwise inherit the top-level ref.
    programme_ref: ProgrammeRefModel | None = None


class RouterDecisionList(BaseModel):
    """Router output: one decision per sub-question (1 for simple queries).

    Backward compatible with the old single RouterDecision in the sense
    that decisions[0] carries the same routing signals a single-decision
    router used to produce.
    """

    intent: Literal[
        "qa",
        "recommendation",
        "comparison"
    ]

    # Programme shared by the whole question (inherited from conversation
    # when the query omits it).
    programme_ref: ProgrammeRefModel | None = None

    decisions: list[RouterSubDecision]

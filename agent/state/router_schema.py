from typing import Literal
from pydantic import BaseModel

class ProgrammeRefModel(BaseModel):
    programme_id: str | None = None
    programme_name: str | None = None

class RouterSubDecision(BaseModel):
    """One sub-question of a (possibly compound) user query.

    intent lives at the top level (RouterDecisionList) -- a compound turn
    is almost always one intent -- so it is intentionally NOT repeated
    here: per-decision intent was the main source of structured-output
    decode instability in the Phase 0 probe.
    """

    retrieval_type: Literal[
        "metadata",
        "section",
        "summary"
    ]

    field: Literal[
        "tuition_fee",
        "deadline",
        "duration",
        "credit",
        "study_mode",
        "entrance_requirement",
        "curriculum",
        None
    ] = None

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

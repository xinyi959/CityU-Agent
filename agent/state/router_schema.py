from typing import Literal
from pydantic import BaseModel

class ProgrammeRefModel(BaseModel):
    programme_id: str | None = None
    programme_name: str | None = None

class RouterDecision(BaseModel):

    intent: Literal[
        "qa",
        "recommendation",
        "comparison"
    ]

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

    programme_ref: ProgrammeRefModel | None = None
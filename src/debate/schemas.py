from typing import Literal

from pydantic import BaseModel, Field


class JudgeResponse(BaseModel):
    winner: Literal["solver_1", "solver_2", "solver_3"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class CritiqueResponse(BaseModel):
    critique: str
    response: str
    accepted: bool


class RefinementResponse(BaseModel):
    changes_made: list[CritiqueResponse]
    refined_solution: str
    refined_answer: str = Field(
        description="Only the final answer, with no explanation, no Markdown, and no full sentence. Examples: '4', '1/6', '6 N'."
    )
    confidence: float = Field(ge=0, le=1)


class SolutionResponse(BaseModel):
    reasoning: str
    answer: str = Field(
        description="Only the final answer, with no explanation, no Markdown, and no full sentence. Examples: '4', '1/6', '6 N'."
    )
    confidence: float = Field(ge=0, le=1)


class RolePreferenceResponse(BaseModel):
    preferred_role: Literal["Solver", "Judge"]
    solver_confidence: float = Field(ge=0, le=1)
    judge_confidence: float = Field(ge=0, le=1)
    reasoning: str
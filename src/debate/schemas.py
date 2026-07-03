from typing import Literal

from pydantic import BaseModel, Field

FINAL_ANSWER_DESCRIPTION = (
    "Only the final answer, with no explanation, no Markdown, and no full "
    "sentence. Examples: '4', '1/6', '6 N'."
)


class JudgeResponse(BaseModel):
    winner: Literal["solver_1", "solver_2", "solver_3"]
    confidence: float
    reasoning: str


class CritiqueResponse(BaseModel):
    critique: str
    response: str
    accepted: bool


class RefinementResponse(BaseModel):
    changes_made: list[CritiqueResponse]
    refined_solution: str
    refined_answer: str = Field(description=FINAL_ANSWER_DESCRIPTION)
    confidence: float


class SolutionResponse(BaseModel):
    reasoning: str = Field(description="why this is right, should be only few work")
    answer: str = Field(description=FINAL_ANSWER_DESCRIPTION)
    confidence: float


class GradeResponse(BaseModel):
    is_correct: bool
    reasoning: str


class RolePreferenceResponse(BaseModel):
    preferred_role: Literal["Solver", "Judge"]
    solver_confidence: float
    judge_confidence: float
    reasoning: str


class DebateResult(BaseModel):
    """Everything one problem's debate produced, keyed by slot / solver id.

    `role_preferences` is keyed by participant slot (model_a..d); `solutions`,
    `peer_reviews`, and `refined_solutions` are keyed by solver id
    (solver_1..3). Peer reviews are raw text (that stage is unstructured).
    """

    problem_id: str
    role_preferences: dict[str, RolePreferenceResponse]
    roles: dict[str, str]
    solutions: dict[str, SolutionResponse]
    peer_reviews: dict[str, list[str]]
    refined_solutions: dict[str, RefinementResponse]
    judgment: JudgeResponse


class EvaluationRecord(BaseModel):
    """One problem's graded outcome: what the debate answered vs the truth."""

    problem_id: str
    answer_type: str
    winner: str
    predicted_answer: str
    correct_answer: str
    is_correct: bool


class RunMetrics(BaseModel):
    """Aggregate metrics for one run.

    `judge_accuracy_on_disagreements` is None when every problem reached full
    solver consensus (no disagreements to judge).
    """

    accuracy: float
    single_llm_accuracy: float
    majority_vote_accuracy: float
    consensus_rate: float
    judge_accuracy_on_disagreements: float | None
    improvement_rate: float

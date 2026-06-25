import math
import re
from typing import Callable, Literal, Optional

from pydantic import BaseModel

# A grader judges whether a free-form predicted answer is equivalent to the
# correct one. Signature: grader(question, predicted, correct) -> bool.
Grader = Callable[[str, str, str], bool]


def _normalize(answer: str) -> str:
    """Lowercase + collapse whitespace for a forgiving string compare."""

    return " ".join(str(answer).strip().lower().split())


def _to_number(answer: str) -> Optional[float]:
    """Extract a single numeric value (int, decimal, or a/b fraction).

    Tolerates trailing units, e.g. "6 N" -> 6.0, "1/6" -> 0.1666..., so a bare
    "6" still matches "6 N". Returns None if no number is found.
    """

    text = str(answer).strip().lower()

    fraction = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", text)
    if fraction:
        numerator, denominator = float(fraction.group(1)), float(fraction.group(2))
        return numerator / denominator if denominator else None

    plain = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(plain.group()) if plain else None


class Problem(BaseModel):
    """One dataset problem plus the logic for grading an answer to it.

    `answer_type` decides how `check()` compares a predicted answer:
    - "number" / "multiple_choice": cheap, deterministic match (no API call).
    - "free_answer": delegate to an LLM `grader` for semantic equivalence.
    """

    id: str
    category: str
    question: str
    correct_answer: str
    answer_type: Literal["number", "multiple_choice", "free_answer"] = "number"
    explanation: str = ""
    difficulty: str = ""

    def check(self, predicted: str, grader: Optional[Grader] = None) -> bool:
        """Return whether `predicted` is a correct answer to this problem."""

        if self.answer_type == "number":
            return self._check_number(predicted)

        if self.answer_type == "multiple_choice":
            return _normalize(predicted) == _normalize(self.correct_answer)

        # free_answer: use the LLM grader, falling back to exact match if none.
        if grader is None:
            return _normalize(predicted) == _normalize(self.correct_answer)

        return grader(self.question, predicted, self.correct_answer)

    def _check_number(self, predicted: str) -> bool:
        predicted_value = _to_number(predicted)
        correct_value = _to_number(self.correct_answer)

        if predicted_value is not None and correct_value is not None:
            # Relative tolerance forgives rounded decimals (0.167 ~ 1/6) while
            # still keeping distinct integers apart.
            return math.isclose(predicted_value, correct_value, rel_tol=1e-3, abs_tol=1e-9)

        # No parseable number on one side -> fall back to string equality.
        return _normalize(predicted) == _normalize(self.correct_answer)

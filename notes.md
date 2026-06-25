# Notes / Known Issues

## Open problems to solve later

### 1. Brittle answer matching (free-text correct answers)
`evaluation.py` uses `exact_match()` = lowercase + strip + exact string equality.
This breaks for problems whose `correct_answer` is a prose sentence rather than a
canonical token.

**Example case:** `logic_006` in `data/problems.json`
- question: "If all Bloops are Razzies and all Razzies are Lazzies, what can be concluded about Bloops?"
- correct_answer: "All Bloops are Lazzies"

A real LLM will likely answer "Bloops are Lazzies", "Every Bloop is a Lazzie",
or add a trailing period — all correct in meaning but scored WRONG by exact match.
Numeric answers ("1/6" vs "0.167", "6 N" vs "6 newtons") have the same problem.

**Impact:** deflates every accuracy / improvement / judge metric once real models
are used. Invisible with the mock client.

**Possible fixes:**
- Normalize answers (numeric/fraction parsing, punctuation stripping), OR
- Use an LLM-judge to check semantic equivalence ("does predicted mean the same as correct?").

Status: TODO

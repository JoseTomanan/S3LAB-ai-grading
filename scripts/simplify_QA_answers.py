"""
Produce dataset/DrawEduMath_SimplifiedQA.csv from DrawEduMath_QA.csv.

For Yes/No-type questions the verbose narrative answer is collapsed to YES or NO.
For open-ended questions the first sentence is kept (trimmed, ≤200 chars).

Usage:
    python scripts/simplify_QA_answers.py
"""

import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_CSV   = REPO_ROOT / "dataset" / "DrawEduMath_QA.csv"
DST_CSV   = REPO_ROOT / "dataset" / "DrawEduMath_SimplifiedQA.csv"

YN_PREFIXES = {"did", "does", "do", "is", "are", "was", "were", "can", "will", "have", "has"}

_NEG = re.compile(
    r"\b(not|n't|none|neither|never|cannot|without|incorrectly|wrongly|failed|fails|missing|forgot|forgets|lacks?|lacking)\b",
    re.IGNORECASE,
)
_FIRST_SENT  = re.compile(r"(?<=[.!?])\s")
_OR_PATTERN  = re.compile(r'\b(\w+)\s+or\s+(\w+)\b', re.IGNORECASE)
_BOOL_WORDS  = {"YES", "NO", "TRUE", "FALSE"}


def _first_sentence(text: str) -> str:
    parts = _FIRST_SENT.split(text, maxsplit=1)
    return parts[0].strip()


def _extract_alternative(question: str, answer: str) -> str | None:
    """For questions that ask 'A or B?', return A, B, or NO from the answer.

    Handles shared-noun patterns like 'straight arrows or curved arrows' where
    the regex naively grabs 'arrows or curved' — if opt_a reappears on the
    right side of 'or', the true distinguishing word is the modifier before it.

    Returns None when no non-boolean alternatives are found so the caller
    can fall back to the normal YES/NO logic.
    """
    match = _OR_PATTERN.search(question)
    if not match:
        return None

    opt_a = match.group(1).upper()
    opt_b = match.group(2).upper()

    # Only handle non-boolean alternatives (skip "yes or no", "true or false")
    if opt_a in _BOOL_WORDS or opt_b in _BOOL_WORDS:
        return None

    # If opt_a (the shared noun) also appears after "or", the real left-side
    # alternative is the modifier word immediately before opt_a in the question.
    # e.g. "straight arrows or curved arrows" → opt_a becomes "STRAIGHT".
    q_before = question[:match.start()]
    q_after  = question[match.end():]
    if re.search(r'\b' + re.escape(match.group(1)) + r'\b', q_after, re.IGNORECASE):
        modifiers = re.findall(r'\b[a-zA-Z]{3,}\b', q_before)
        if modifiers:
            opt_a = modifiers[-1].upper()

    # Negation → the student used neither option
    if _NEG.search(answer):
        return "NO"

    ans_upper = answer.upper()

    def has_word(w: str) -> bool:
        return bool(re.search(r'\b' + re.escape(w) + r'\b', ans_upper))

    has_a = has_word(opt_a)
    has_b = has_word(opt_b)
    if has_a and not has_b:
        return opt_a
    if has_b and not has_a:
        return opt_b
    if has_a:  # both present — return whichever appears first
        return opt_a if ans_upper.index(opt_a) <= ans_upper.index(opt_b) else opt_b
    return "NO"


def simplify(question: str, answer: str) -> str:
    first_word = re.split(r"\W+", question.strip().lower())[0]

    if first_word in YN_PREFIXES:
        # Questions like "Did the student use a horizontal or vertical number line?"
        # embed specific alternatives — extract the actual value instead of YES/NO.
        alt = _extract_alternative(question, answer)
        if alt is not None:
            return alt

        sent = _first_sentence(answer)
        low  = sent.lstrip().lower()
        if low.startswith("yes"):
            return "YES"
        if low.startswith("no"):
            return "NO"
        if _NEG.search(sent):
            return "NO"
        return "YES"

    # Open-ended: return first sentence, strip trailing period, cap length
    sent = _first_sentence(answer).rstrip(".")
    if len(sent) > 200:
        cut = sent[:200].rsplit(" ", 1)[0]
        sent = cut + "…"
    return sent


# ---------------------------------------------------------------------------
# Sanity fixtures — run before touching the CSV
# ---------------------------------------------------------------------------
SANITY_CASES = [
    (
        "Did students label the number line correctly?",
        "The student labeled the number line correctly.",
        "YES",
    ),
    (
        "Does the student label the magnitude of each arrow?",
        "The student does not label the magnitude of each arrow.",
        "NO",
    ),
    (
        "Is the final answer clearly marked on the number line?",
        "Yes, the final answer clearly marked on the number line; the student circled 2.",
        "YES",
    ),
    (
        "Did the student label the magnitude of each arrow on the number line diagram?",
        "The student did not label the magnitude of each arrow on the number line diagram.",
        "NO",
    ),
    (
        "What errors does the student make in their response? If there are none, write that there is no error",
        "There is no error in the student response.",
        "There is no error in the student response",
    ),
    (
        "What strategy does the student use to solve the problem?",
        "The student uses a number line modeling strategy to solve the problem. "
        "This strategy includes drawing a number line, labeling the tick marks, "
        "and drawing an arrow to correspond with each integer in the problem.",
        "The student uses a number line modeling strategy to solve the problem",
    ),
    # Alternative-extraction cases
    (
        "Did the student use a horizontal or vertical number line to model this problem?",
        "The student used a horizontal number line to model this problem.",
        "HORIZONTAL",
    ),
    (
        "Did the student draw a horizontal or vertical number line diagram?",
        "The student drew a vertical number line diagram.",
        "VERTICAL",
    ),
    (
        "Did the student use a horizontal or vertical number line to model this problem?",
        "The student did not use a number line, horizontal or vertical, to model this problem.",
        "NO",
    ),
    (
        "Did the student use a horizontal or vertical number line to model this problem?",
        "The student did not use a horizontal number line to model this problem.",
        "NO",
    ),
    (
        "Did the student use straight arrows or curved arrows to represent the sum?",
        "The student used straight arrows to represent the sum.",
        "STRAIGHT",
    ),
]


def _run_sanity() -> None:
    for q, a, expected in SANITY_CASES:
        got = simplify(q, a)
        assert got == expected, (
            f"\nsimplify({q!r}, ...)\n  got:      {got!r}\n  expected: {expected!r}"
        )
    print(f"Sanity: {len(SANITY_CASES)} cases passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _run_sanity()

    df = pd.read_csv(SRC_CSV)
    out_rows = []
    yes_n = no_n = open_n = 0

    for _, row in df.iterrows():
        try:
            qa_items = json.loads(row["QA Teacher"])
        except (json.JSONDecodeError, KeyError):
            qa_items = []

        simplified = []
        for item in qa_items:
            q = item["question"].strip()
            a = simplify(q, item["answer"].strip())
            simplified.append({"question": q, "answer": a})
            if   a == "YES": yes_n += 1
            elif a == "NO":  no_n  += 1
            else:            open_n += 1

        out_rows.append({
            "Problem ID": row["Problem ID"],
            "Image Name": row["Image Name"],
            "QA Teacher": json.dumps(simplified, ensure_ascii=False),
        })

    pd.DataFrame(out_rows).to_csv(DST_CSV, index=False, encoding="utf-8")
    print(f"Wrote {len(out_rows)} rows -> {DST_CSV}")
    print(f"  YES={yes_n}  NO={no_n}  open-ended={open_n}")


if __name__ == "__main__":
    main()

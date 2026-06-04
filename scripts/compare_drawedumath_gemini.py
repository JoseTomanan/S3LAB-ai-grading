"""
Like-for-like comparison: OUR pipeline vs DrawEduMath's PRE-EXISTING model
answers, over the EXACT same (image, question) items our pipeline scored.

Our pipeline's results live in an existing report table, e.g.
  dataset/DrawEduMath/LLM_evaluations_QA_Teacher_KS1_ALL.csv
which contains 1396 YES/NO test rows (row_num, problem_id, image_name,
question_idx, question, model_answer, ground_truth, correct).

This script takes that exact set of items and, for each one, looks up the
PRE-EXISTING answer from DrawEduMath's published model answers
(`<Model>_Answer-TeacherQ` in DrawEduMath_QA_with_Model_Answers.csv) and
scores it against the same ground truth. NO API/model calls are made.

The output is a single side-by-side comparison CSV over the identical item
set, so "ours vs theirs" is a true apples-to-apples comparison.

Usage:
    python scripts/compare_drawedumath_gemini.py
    python scripts/compare_drawedumath_gemini.py --model Gemini   # or Claude/GPT4/Llama
    python scripts/compare_drawedumath_gemini.py --ours dataset/DrawEduMath/LLM_evaluations_QA_Teacher_first_1000_KS1.csv
"""

import argparse
import json
import re
from difflib import get_close_matches
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ANS = REPO_ROOT / "dataset" / "DrawEduMath" / "DrawEduMath_QA_with_Model_Answers.csv"
DEFAULT_OURS = REPO_ROOT / "dataset" / "DrawEduMath" / "LLM_evaluations_QA_Teacher_KS1_ALL.csv"
DEFAULT_OUT = REPO_ROOT / "dataset" / "DrawEduMath" / "LLM_evaluations_QA_Teacher_OursVsGemini_KS1_ALL.csv"


def normalize_q(q: str) -> str:
    """Lowercase, strip, collapse whitespace — for matching question text."""
    return re.sub(r"\s+", " ", str(q).strip().lower())


def to_yes_no(ans: str) -> str:
    """
    Reduce a free-form model answer to YES / NO when possible.

    DrawEduMath's published answers are prose like "Yes. \\n",
    "No, they are reversed. \\n", or "Straight arrows.". We map a leading
    yes/no token to YES/NO; anything else is uppercased and trimmed so it
    fails the == comparison against the YES/NO ground truth (counting as
    incorrect, mirroring how our own pipeline treats answers like
    "STRAIGHT ARROWS").
    """
    cleaned = ans.strip().lstrip("*->•\"' \t")
    upper = cleaned.upper()
    if re.match(r"^YES\b", upper):
        return "YES"
    if re.match(r"^NO\b", upper):
        return "NO"
    return upper.rstrip(". \n").strip()


def read_ours_table(path: Path) -> pd.DataFrame:
    """Read an existing KS1-style report, skipping its leading metrics block."""
    with path.open(encoding="utf-8") as f:
        lines = f.readlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("row_num,"))
    return pd.read_csv(path, skiprows=hdr)


def build_ref_index(model_ans_path: Path, answer_col: str) -> dict:
    """Map (ProblemID, ImageID) -> {normalized_question: raw answer}."""
    ma = pd.read_csv(model_ans_path)
    index: dict[tuple, dict[str, str]] = {}
    for _, row in ma.iterrows():
        per_image: dict[str, str] = {}
        try:
            for it in json.loads(row[answer_col]):
                per_image[normalize_q(it["question"])] = it["answer"]
        except (json.JSONDecodeError, TypeError):
            pass
        index[(row["ProblemID"], row["ImageID"])] = per_image
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare our pipeline vs pre-existing DrawEduMath answers on identical items")
    parser.add_argument("--model", default="Gemini",
                        choices=["Gemini", "Claude", "GPT4", "Llama"],
                        help="Which pre-existing model's TeacherQ answers to score against ours")
    parser.add_argument("--ours", type=Path, default=DEFAULT_OURS,
                        help="Our existing KS1-style report defining the item set")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    answer_col = f"{args.model}_Answer-TeacherQ"
    ours = read_ours_table(args.ours)
    ref_index = build_ref_index(MODEL_ANS, answer_col)

    records: list[dict] = []
    n = len(ours)
    our_correct = ref_correct = agree = n_missing = n_unparseable = 0

    for _, r in ours.iterrows():
        pid, img = r["problem_id"], r["image_name"]
        question = r["question"]
        gt = str(r["ground_truth"]).strip()
        our_ans = str(r["model_answer"]).strip()
        our_ok = bool(r["correct"])

        per_image = ref_index.get((pid, img), {})
        key = normalize_q(question)
        raw = per_image.get(key)
        if raw is None and per_image:
            close = get_close_matches(key, list(per_image.keys()), n=1, cutoff=0.85)
            if close:
                raw = per_image[close[0]]

        if raw is None:
            ref_ans = "MISSING"
            n_missing += 1
        else:
            ref_ans = to_yes_no(raw)
            if ref_ans not in ("YES", "NO"):
                n_unparseable += 1

        ref_ok = ref_ans == gt
        our_correct += int(our_ok)
        ref_correct += int(ref_ok)
        agree += int(our_ans == ref_ans)

        records.append({
            "row_num": r["row_num"],
            "problem_id": pid,
            "image_name": img,
            "question_idx": r["question_idx"],
            "question": question,
            "ground_truth": gt,
            "our_answer": our_ans,
            "our_correct": our_ok,
            "ref_answer": ref_ans,
            "ref_correct": ref_ok,
        })

    table = pd.DataFrame.from_records(records)
    our_acc = our_correct / n * 100 if n else 0.0
    ref_acc = ref_correct / n * 100 if n else 0.0
    agree_pct = agree / n * 100 if n else 0.0

    with args.output.open("w", encoding="utf-8", newline="") as f:
        f.write("Metric,Value\n")
        f.write(f"Number of tests,{n}\n")
        f.write(f"Our pipeline correct,{our_correct}\n")
        f.write(f"Our pipeline accuracy,{our_acc:.2f}%\n")
        f.write(f"{args.model} (pre-existing) correct,{ref_correct}\n")
        f.write(f"{args.model} (pre-existing) accuracy,{ref_acc:.2f}%\n")
        f.write(f"Answer agreement,{agree} ({agree_pct:.2f}%)\n")
        f.write(f"{args.model} missing answers,{n_missing}\n")
        f.write(f"{args.model} non-YES/NO answers,{n_unparseable}\n")
        f.write("\n")
    table.to_csv(args.output, mode="a", index=False)

    print(f"Wrote {args.output}")
    print(f"  Items compared:        {n}")
    print(f"  Our pipeline correct:  {our_correct} ({our_acc:.2f}%)")
    print(f"  {args.model} pre-existing: {ref_correct} ({ref_acc:.2f}%)")
    print(f"  Agreement:             {agree} ({agree_pct:.2f}%)")
    print(f"  {args.model} missing: {n_missing}   non-YES/NO: {n_unparseable}")


if __name__ == "__main__":
    main()

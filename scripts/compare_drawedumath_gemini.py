"""
Like-for-like comparison: OUR pipeline vs DrawEduMath's PRE-EXISTING model
answers, over the EXACT same (image, question) items our pipeline scored.

Our pipeline's results live in an existing report table, e.g.
  dataset/DrawEduMath/LLM_evaluations_QA_Teacher_KS1_ALL.csv
which contains 1396 YES/NO test rows (row_num, problem_id, image_name,
question_idx, question, model_answer, ground_truth, correct).

This script takes that exact set of items and, for each one, looks up the
PRE-EXISTING answer from a published DrawEduMath model-answers file and
scores it against the same ground truth. NO API/model calls are made.

Two pre-existing sources are supported (pick with --source):
  * aftermath — (default) the follow-up paper's long-format aftermath_predictions.csv
               (one row per model/question). Carries 2025-era models, including the
               exact label "Gemini 2.5 Pro". Free-form prose answers, scored by the
               paper's ensemble judge (the "Score" column, 1 = correct).
  * legacy   — the original benchmark file DrawEduMath_QA_with_Model_Answers.csv
               (wide/nested: answers packed as JSON in `<Model>_Answer-TeacherQ`).
               Models: Gemini (1.5 Pro), Claude (3.5 Sonnet), GPT4 (4o), Llama.
               Terse answers reduced to YES/NO and matched against ground truth.

The output is a single side-by-side comparison CSV over the identical item
set, so "ours vs theirs" is a true apples-to-apples comparison.

Both source CSVs are large and git-ignored; download aftermath_predictions.csv with:
    curl -sL -o dataset/DrawEduMath/aftermath_predictions.csv \\
        https://huggingface.co/datasets/lucy3/aftermath_predictions/resolve/main/predictions.csv

Usage:
    python scripts/compare_drawedumath_gemini.py                                    # aftermath Gemini 2.5 Pro
    python scripts/compare_drawedumath_gemini.py --model "Claude Sonnet 4.5"        # or "GPT-5", etc.
    python scripts/compare_drawedumath_gemini.py --source legacy --model Claude     # original benchmark
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
# Long-format predictions from the follow-up paper "The Aftermath of DrawEduMath"
# (HF dataset lucy3/aftermath_predictions). Carries newer models incl. "Gemini 2.5 Pro".
AFTERMATH = REPO_ROOT / "dataset" / "DrawEduMath" / "aftermath_predictions.csv"
DEFAULT_OURS = REPO_ROOT / "dataset" / "DrawEduMath" / "LLM_evaluations_QA_Teacher_KS1_ALL.csv"


def out_path_for(model: str) -> Path:
    """Default report path, e.g. ...OursVsGemini2.5Pro_KS1_ALL.csv."""
    token = re.sub(r"[^0-9A-Za-z.]+", "", model)
    return REPO_ROOT / "dataset" / "DrawEduMath" / f"LLM_evaluations_QA_Teacher_OursVs{token}_KS1_ALL.csv"


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


def build_ref_index_aftermath(path: Path, model_name: str, qa_type: str = "teacher") -> dict:
    """
    Map (ProblemID, ImageID) -> {normalized_question: {"answer", "score"}} from
    the long-format aftermath_predictions.csv (one row per model/image/question).

    Unlike the original benchmark, these answers are free-form chain-of-thought
    prose, so they cannot be reduced to YES/NO. Instead the follow-up paper ships
    a per-answer correctness judgement in the "Score" column (1 = correct,
    0 = incorrect, -1 = unscored), produced by an ensemble judge against the
    teacher's reference answer. We carry that Score through for scoring.

    Columns: "Model Name", "Problem ID", "QA Type", "Image Name", "Question",
    "Model Answer", "Score". We keep only the requested model and QA type.
    """
    df = pd.read_csv(path)
    df = df[(df["Model Name"] == model_name) & (df["QA Type"] == qa_type)]
    if df.empty:
        raise SystemExit(f"No rows for Model Name={model_name!r}, QA Type={qa_type!r} in {path.name}")
    index: dict[tuple, dict[str, dict]] = {}
    for _, row in df.iterrows():
        index.setdefault((row["Problem ID"], row["Image Name"]), {})[
            normalize_q(row["Question"])] = {"answer": row["Model Answer"], "score": int(row["Score"])}
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare our pipeline vs pre-existing DrawEduMath answers on identical items")
    parser.add_argument("--source", default="aftermath", choices=["legacy", "aftermath"],
                        help="Which pre-existing answer file to score against ours (default aftermath)")
    parser.add_argument("--model", default=None,
                        help="Reference model. legacy: Gemini/Claude/GPT4/Llama (default Gemini). "
                             "aftermath: exact 'Model Name' value, e.g. 'Gemini 2.5 Pro' (default).")
    parser.add_argument("--ours", type=Path, default=DEFAULT_OURS,
                        help="Our existing KS1-style report defining the item set")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV (default derived from --model)")
    args = parser.parse_args()

    if args.source == "aftermath":
        model = args.model or "Gemini 2.5 Pro"
        ref_index = build_ref_index_aftermath(AFTERMATH, model)
    else:
        model = args.model or "Gemini"
        ref_index = build_ref_index(MODEL_ANS, f"{model}_Answer-TeacherQ")
    output = args.output or out_path_for(model)
    is_aftermath = args.source == "aftermath"
    ours = read_ours_table(args.ours)

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

        if is_aftermath:
            # Free-form prose graded by the follow-up paper's ensemble judge.
            if raw is None:
                ref_ans, ref_ok = "MISSING", False
                n_missing += 1
            else:
                ref_ans = re.sub(r"\s+", " ", str(raw["answer"]).strip())[:300]
                ref_ok = raw["score"] == 1
        else:
            # Original benchmark: terse answers reduced to YES/NO and matched.
            if raw is None:
                ref_ans = "MISSING"
                n_missing += 1
            else:
                ref_ans = to_yes_no(raw)
                if ref_ans not in ("YES", "NO"):
                    n_unparseable += 1
            ref_ok = ref_ans == gt
            agree += int(our_ans == ref_ans)

        our_correct += int(our_ok)
        ref_correct += int(ref_ok)

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

    with output.open("w", encoding="utf-8", newline="") as f:
        f.write("Metric,Value\n")
        f.write(f"Number of tests,{n}\n")
        f.write(f"Our pipeline correct,{our_correct}\n")
        f.write(f"Our pipeline accuracy,{our_acc:.2f}%\n")
        f.write(f"{model} (pre-existing) correct,{ref_correct}\n")
        f.write(f"{model} (pre-existing) accuracy,{ref_acc:.2f}%\n")
        if is_aftermath:
            f.write(f"{model} scoring,ensemble judge (Score==1)\n")
            f.write(f"{model} missing answers,{n_missing}\n")
        else:
            agree_pct = agree / n * 100 if n else 0.0
            f.write(f"Answer agreement,{agree} ({agree_pct:.2f}%)\n")
            f.write(f"{model} missing answers,{n_missing}\n")
            f.write(f"{model} non-YES/NO answers,{n_unparseable}\n")
        f.write("\n")
    table.to_csv(output, mode="a", index=False)

    print(f"Wrote {output}")
    print(f"  Items compared:        {n}")
    print(f"  Our pipeline correct:  {our_correct} ({our_acc:.2f}%)")
    print(f"  {model} pre-existing: {ref_correct} ({ref_acc:.2f}%)")
    if is_aftermath:
        print(f"  Scoring: ensemble judge (Score==1)   missing: {n_missing}")
    else:
        print(f"  Agreement:             {agree} ({agree / n * 100:.2f}%)")
        print(f"  {model} missing: {n_missing}   non-YES/NO: {n_unparseable}")


if __name__ == "__main__":
    main()

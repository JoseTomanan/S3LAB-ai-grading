"""
Compare LLM_evaluations_QA_Teacher.jsonl against DrawEduMath_SimplifiedQA.csv
and write an accuracy report to scripts/LLM_EVALUATION_REPORT.csv.

Each question is counted as one test. Answers are compared after
case-insensitive normalization (lowercase, collapsed whitespace, stripped
trailing punctuation).

Usage:
    python scripts/evaluate_LLM_accuracy.py
    python scripts/evaluate_LLM_accuracy.py --input <jsonl> --output <csv>
"""

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT   = Path(__file__).resolve().parent.parent
DEFAULT_IN  = REPO_ROOT / "dataset" / "DrawEduMath" / "LLM_evaluations_QA_Teacher.jsonl"
GT_CSV      = REPO_ROOT / "dataset" / "DrawEduMath_SimplifiedQA.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "LLM_EVALUATION_REPORT.csv"

_WS          = re.compile(r"\s+")
_TRAIL_PUNCT = ".,;:!?"


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").strip().lower()
    s = _WS.sub(" ", s).rstrip(_TRAIL_PUNCT).strip()
    return s


def is_correct(model: str, truth: str) -> bool:
    return normalize(model) == normalize(truth)


@dataclass
class ResultRow:
    row_num:      int
    problem_id:   int
    image_name:   str
    question_idx: int
    question:     str
    model_answer: str
    ground_truth: str
    correct:      bool


def load_ground_truth(gt_csv: Path) -> dict[int, list[dict]]:
    df = pd.read_csv(gt_csv)
    out: dict[int, list[dict]] = {}
    for idx, row in df.iterrows():
        row_num = int(idx) + 1 # type: ignore
        try:
            out[row_num] = json.loads(row["QA Teacher"])
        except (json.JSONDecodeError, KeyError):
            out[row_num] = []
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM answer accuracy")
    parser.add_argument("--input",  type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: JSONL not found: {args.input}")
        return

    gt_by_row = load_ground_truth(GT_CSV)

    results: list[ResultRow] = []
    skipped_rows = 0

    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            row_num    = rec["row_num"]
            problem_id = rec["problem_id"]
            image_name = rec["image_name"]
            questions  = rec["questions"]
            parse_ok   = rec.get("parse_ok", False)
            raw        = rec.get("raw_response")
            m_answers  = rec.get("model_answers", [])

            gt_items = gt_by_row.get(row_num, [])

            if not gt_items:
                print(f"WARNING row={row_num}: no ground-truth found — skipping row.")
                skipped_rows += 1
                continue

            if len(gt_items) != len(questions):
                print(
                    f"WARNING row={row_num}: GT has {len(gt_items)} questions "
                    f"but JSONL has {len(questions)} — skipping row."
                )
                skipped_rows += 1
                continue

            for q_idx, question in enumerate(questions):
                gt_answer = gt_items[q_idx]["answer"]

                if not parse_ok or raw is None or q_idx >= len(m_answers):
                    model_answer = ""
                    correct      = False
                else:
                    model_answer = m_answers[q_idx]
                    correct      = is_correct(model_answer, gt_answer)

                results.append(ResultRow(
                    row_num      = row_num,
                    problem_id   = problem_id,
                    image_name   = image_name,
                    question_idx = q_idx,
                    question     = question,
                    model_answer = model_answer,
                    ground_truth = gt_answer,
                    correct      = correct,
                ))

    total     = len(results)
    correct_n = sum(1 for r in results if r.correct)
    accuracy  = correct_n / total if total else 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["Number of tests",  total])
        w.writerow(["Number correct",   correct_n])
        w.writerow(["Accuracy rate",    f"{accuracy:.2%}"])
        if skipped_rows:
            w.writerow(["Skipped rows (alignment mismatch)", skipped_rows])
        w.writerow([])
        w.writerow(["row_num", "problem_id", "image_name", "question_idx",
                    "question", "model_answer", "ground_truth", "correct"])
        for r in results:
            w.writerow([r.row_num, r.problem_id, r.image_name, r.question_idx,
                        r.question, r.model_answer, r.ground_truth, r.correct])

    print(f"Tests   : {total}")
    print(f"Correct : {correct_n}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Report  -> {args.output}")


if __name__ == "__main__":
    main()

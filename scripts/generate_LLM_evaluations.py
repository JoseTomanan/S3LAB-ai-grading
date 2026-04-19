"""
Evaluate DrawEduMath student-answer images (Claude_Postprocessing/)
against the QA Teacher rubric questions from DrawEduMath_QA.csv.

Usage:
    python scripts/generate_LLM_evaluations.py               # all 2030 rows
    python scripts/generate_LLM_evaluations.py --start 8 --end 8
    python scripts/generate_LLM_evaluations.py --dry-run --start 8 --end 8
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT   = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")

from logic.ai_interface import AIAnswerEvaluator, ANSWER_RUBRIC_PROMPT  # pyright: ignore[reportMissingImports] E402
from logic.box_segmenter import BoxSegmenter  # pyright: ignore[reportMissingImports] E402


CSV_PATH    = REPO_ROOT / "dataset" / "DrawEduMath_SimplifiedQA.csv"
IMAGES_DIR  = REPO_ROOT / "dataset" / "DrawEduMath" / "Claude_Postprocessing"
DEFAULT_OUT = REPO_ROOT / "dataset" / "DrawEduMath" / "LLM_evaluations_QA_Teacher.jsonl"


def load_done(output_path: Path) -> set[int]:
    done: set[int] = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["row_num"])
                except Exception:
                    pass
    return done


def build_prompt(questions: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {q.strip()}" for i, q in enumerate(questions))
    return (
        f"{ANSWER_RUBRIC_PROMPT}\n"
        "Return your answers as a JSON array of strings, one element per question, "
        "in the same order as the questions below.\n"
        f"QUESTION:\nPROMPT:\n{numbered}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM evaluations for DrawEduMath")
    parser.add_argument("--qa-column", default="QA Teacher", help="CSV column to read QA from")
    parser.add_argument("--start", type=int, default=1,    help="First row (1-indexed, inclusive)")
    parser.add_argument("--end",   type=int, default=2030, help="Last row  (1-indexed, inclusive)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the prompt for the first row and exit without calling Gemini")
    args = parser.parse_args()

    df    = pd.read_csv(CSV_PATH)
    total = args.end - args.start + 1

    done_rows: set[int] = set() if args.dry_run else load_done(args.output)
    if done_rows:
        print(f"Resuming — {len(done_rows)} rows already done.")

    evaluator     = None if args.dry_run else AIAnswerEvaluator() # pyright: ignore[reportUnboundVariable] E0602
    box_segmenter = None if args.dry_run else BoxSegmenter()
    out_file  = None if args.dry_run else open(args.output, "a", encoding="utf-8")

    try:
        completed = 0
        for row_num in range(args.start, args.end + 1):
            if row_num in done_rows:
                completed += 1
                continue

            idx = row_num - 1
            if idx >= len(df):
                print(f"Row {row_num}: beyond CSV length, stopping.")
                break

            row = df.iloc[idx]

            # Resolve local image — extension comes from the CSV Image Name
            ext         = row["Image Name"].split(".")[-1].lower()
            local_image = f"{row_num}.{ext}"
            image_path  = IMAGES_DIR / local_image

            if not image_path.exists():
                print(f"[{completed+1}/{total}] row={row_num}: SKIP — {local_image} not found")
                completed += 1
                continue

            try:
                qa_items = json.loads(row[args.qa_column])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[{completed+1}/{total}] row={row_num}: SKIP — QA parse error: {e}")
                completed += 1
                continue

            yn_items = [
                (item["question"].strip(), item["answer"].strip())
                for item in qa_items
                if item["answer"].strip() in ("YES", "NO")
            ]
            if not yn_items:
                print(f"[{completed+1}/{total}] row={row_num}: SKIP — no YES/NO questions")
                completed += 1
                continue

            questions    = [q for q, _ in yn_items]
            ground_truth = [a for _, a in yn_items]

            prompt = build_prompt(questions)

            if args.dry_run:
                print(f"=== Dry run: row {row_num} — {local_image} ===")
                print(f"Image: {image_path} (would apply beautify_scan enhancement)")
                print(f"\n--- Prompt ({len(questions)} questions) ---\n{prompt}")
                print("\n--- Ground truth ---")
                for q, a in zip(questions, ground_truth):
                    print(f"  Q: {q}")
                    print(f"  A: {a}")
                return

            image_bytes = box_segmenter.beautify_scan(image_path.read_bytes()) # pyright: ignore[reportOptionalMemberAccess] E0602
            raw_response: str | None = evaluator._send_image_prompt(image_bytes, prompt, response_schema=list[str]) # pyright: ignore[reportOptionalMemberAccess] E0602

            if raw_response is None:
                model_answers: list[str] = []
                parse_ok = False
            else:
                try:
                    parsed = json.loads(raw_response)
                    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                        model_answers = [a.strip().upper() for a in parsed]
                    else:
                        model_answers = [str(x).strip().upper() for x in parsed] if isinstance(parsed, list) else []
                except json.JSONDecodeError:
                    model_answers = []
                parse_ok = len(model_answers) == len(questions)

            comparisons = (
                [m == g for m, g in zip(model_answers, ground_truth)]
                if parse_ok else []
            )

            record = {
                "row_num":              row_num,
                "problem_id":           int(row["Problem ID"]),
                "image_name":           row["Image Name"],
                "local_image":          local_image,
                "questions":            questions,
                "ground_truth_answers": ground_truth,
                "model_answers":        model_answers,
                "comparisons":          comparisons,
                "raw_response":         raw_response,
                "parse_ok":             parse_ok,
            }

            assert out_file is not None
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()

            completed += 1
            print(f"[{completed}/{total}] row={row_num} parse_ok={parse_ok}")
    finally:
        if out_file:
            out_file.close()


if __name__ == "__main__":
    main()

"""
Evaluate DrawEduMath student-answer images (Claude_Postprocessing/)
against the QA Teacher rubric questions from DrawEduMath_QA.csv.

Usage:
    python scripts/generate_LLM_evaluations.py               # all 2030 rows
    python scripts/generate_LLM_evaluations.py --start 8 --end 8
    python scripts/generate_LLM_evaluations.py --dry-run --start 8 --end 8
    python scripts/generate_LLM_evaluations.py --workers 8   # more parallelism
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

_thread_local = threading.local()


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


def _local_instances() -> tuple[AIAnswerEvaluator, BoxSegmenter]:
    """Return (evaluator, box_segmenter) local to the calling thread."""
    if not hasattr(_thread_local, "evaluator"):
        _thread_local.evaluator    = AIAnswerEvaluator()
        _thread_local.box_segmenter = BoxSegmenter()
    return _thread_local.evaluator, _thread_local.box_segmenter


def process_row(
    row_num: int,
    df: pd.DataFrame,
    qa_column: str,
) -> tuple[int, dict | None, str | None]:
    """
    Returns (row_num, record, skip_reason).
    record is None for skipped rows; skip_reason is None for processed rows.
    """
    row = df.iloc[row_num - 1]

    ext         = row["Image Name"].split(".")[-1].lower()
    local_image = f"{row_num}.{ext}"
    image_path  = IMAGES_DIR / local_image

    if not image_path.exists():
        return row_num, None, f"SKIP — {local_image} not found"

    try:
        qa_items = json.loads(row[qa_column])
    except (json.JSONDecodeError, KeyError) as e:
        return row_num, None, f"SKIP — QA parse error: {e}"

    yn_items = [
        (item["question"].strip(), item["answer"].strip())
        for item in qa_items
        if item["answer"].strip() in ("YES", "NO")
    ]
    if not yn_items:
        return row_num, None, "SKIP — no YES/NO questions"

    questions    = [q for q, _ in yn_items]
    ground_truth = [a for _, a in yn_items]
    prompt       = build_prompt(questions)

    evaluator, box_segmenter = _local_instances()
    image_bytes = box_segmenter.beautify_scan(image_path.read_bytes())
    t0 = time.perf_counter()
    raw_response: str | None = evaluator._send_image_prompt(image_bytes, prompt, response_schema=list[str])  # pyright: ignore[reportPrivateUsage]
    elapsed_s = round(time.perf_counter() - t0, 3)

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
        "elapsed_s":            elapsed_s,
    }

    return row_num, record, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM evaluations for DrawEduMath")
    parser.add_argument("--qa-column", default="QA Teacher", help="CSV column to read QA from")
    parser.add_argument("--start",   type=int,  default=1,    help="First row (1-indexed, inclusive)")
    parser.add_argument("--end",     type=int,  default=2030, help="Last row  (1-indexed, inclusive)")
    parser.add_argument("--workers", type=int,  default=4,    help="Number of parallel API threads")
    parser.add_argument("--output",  type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the prompt for the first row and exit without calling the API")
    args = parser.parse_args()

    df    = pd.read_csv(CSV_PATH)
    total = args.end - args.start + 1

    done_rows: set[int] = set() if args.dry_run else load_done(args.output)
    if done_rows:
        print(f"Resuming — {len(done_rows)} rows already done.")

    # Build pending list, stopping at the first row that exceeds CSV bounds.
    pending: list[int] = []
    for rn in range(args.start, args.end + 1):
        if rn - 1 >= len(df):
            print(f"Row {rn}: beyond CSV length, stopping.")
            break
        if rn not in done_rows:
            pending.append(rn)

    # ── dry-run: print first processable row and exit ────────────────────────
    if args.dry_run:
        for row_num in pending:
            row         = df.iloc[row_num - 1]
            ext         = row["Image Name"].split(".")[-1].lower()
            local_image = f"{row_num}.{ext}"
            image_path  = IMAGES_DIR / local_image

            try:
                qa_items = json.loads(row[args.qa_column])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"row={row_num}: SKIP — QA parse error: {e}")
                continue

            yn_items = [
                (item["question"].strip(), item["answer"].strip())
                for item in qa_items
                if item["answer"].strip() in ("YES", "NO")
            ]
            if not yn_items:
                print(f"row={row_num}: SKIP — no YES/NO questions")
                continue

            questions    = [q for q, _ in yn_items]
            ground_truth = [a for _, a in yn_items]
            prompt       = build_prompt(questions)

            print(f"=== Dry run: row {row_num} — {local_image} ===")
            print(f"Image: {image_path} (would apply beautify_scan enhancement)")
            print(f"\n--- Prompt ({len(questions)} questions) ---\n{prompt}")
            print("\n--- Ground truth ---")
            for q, a in zip(questions, ground_truth):
                print(f"  Q: {q}")
                print(f"  A: {a}")
            return
        return

    # ── threaded processing ───────────────────────────────────────────────────
    completed  = total - len(pending)  # already-done + out-of-range rows
    write_lock = threading.Lock()

    with open(args.output, "a", encoding="utf-8") as out_file:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_row, rn, df, args.qa_column): rn
                for rn in pending
            }
            for future in as_completed(futures):
                row_num, record, skip_reason = future.result()
                with write_lock:
                    completed += 1
                    if skip_reason:
                        print(f"[{completed}/{total}] row={row_num}: {skip_reason}")
                    else:
                        assert record is not None
                        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_file.flush()
                        print(f"[{completed}/{total}] row={row_num} parse_ok={record['parse_ok']} elapsed={record['elapsed_s']}s")


if __name__ == "__main__":
    main()

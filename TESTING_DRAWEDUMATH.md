# DrawEduMath Functionality Testing Guide

This guide covers how to use the DrawEduMath dataset to measure the accuracy of the Gemini AI evaluator against teacher-annotated ground truth.

---

## Overview

**DrawEduMath** is a dataset of ~2,030 scanned handwritten student answer images for elementary-level number-line addition problems. Each image has a corresponding set of YES/NO rubric questions answered by a teacher (the ground truth).

The testing workflow has three stages:

```
1. generate_LLM_evaluations.py   ← send images to Gemini, write results to JSONL
2. evaluate_LLM_accuracy.py      ← compare JSONL results against teacher answers → CSV
3. analyze_evaluations.py        ← deep statistical analysis of accuracy CSV
```

Each stage can be run independently if you already have the output from the previous one.

---

## Prerequisites

- Python 3.10+ with a virtual environment (see `INSTALL.md`)
- Root-level `.env` containing `GEMINI_API_KEY` (required for Stage 1 only)
- Dependencies installed:

```bash
pip install -r requirements.txt          # root requirements
pip install -r app/backend/requirements.txt
```

All scripts must be run from the **repository root**.

---

## Dataset Structure

```
dataset/
├── DrawEduMath_QA.csv                # Full multi-evaluator QA data (5,443 rows)
├── DrawEduMath_SimplifiedQA.csv      # YES/NO teacher answers only — the ground truth (2,031 rows)
├── DrawEduMath_QA_ImageURL.txt       # AWS S3 URLs for original images
└── DrawEduMath/
    ├── Claude_Postprocessing/        # ~1,900 local images named {row_number}.{jpg,png}
    └── LLM_evaluations_*.jsonl       # Pre-computed evaluation batches (already present)
```

**`DrawEduMath_SimplifiedQA.csv` columns:**

| Column | Description |
|--------|-------------|
| Problem ID | Unique problem identifier |
| Image Name | Original image filename |
| Image SHA256 | Image hash for verification |
| QA Teacher | JSON array of YES/NO question-answer pairs (ground truth) |

---

## Stage 1 — Generate LLM Evaluations

`scripts/generate_LLM_evaluations.py` sends each student-answer image to Gemini along with the rubric questions from the teacher QA column, then writes the model's YES/NO responses to a JSONL file.

### Basic Usage

```bash
# Dry-run: print the prompt for row 1 without calling the API
python scripts/generate_LLM_evaluations.py --dry-run --start 1 --end 1

# Process a small batch (rows 1–50)
python scripts/generate_LLM_evaluations.py --start 1 --end 50

# Process the full dataset (2,030 rows) with more parallelism
python scripts/generate_LLM_evaluations.py --workers 8

# Resume or append to an existing output file (already-processed rows are skipped)
python scripts/generate_LLM_evaluations.py --start 501 --end 1000 \
  --output dataset/DrawEduMath/LLM_evaluations_QA_Teacher_0501_1000.jsonl
```

### All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--start N` | `1` | First row to process (1-indexed, inclusive) |
| `--end N` | `2030` | Last row to process (1-indexed, inclusive) |
| `--ranges "A-B,C-D"` | — | Comma-separated ranges; overrides `--start`/`--end` |
| `--workers N` | `4` | Parallel API threads |
| `--qa-column NAME` | `"QA Teacher"` | CSV column to read rubric questions from |
| `--output PATH` | `dataset/DrawEduMath/LLM_evaluations_QA_Teacher.jsonl` | Output file |
| `--dry-run` | — | Print the prompt for the first processable row and exit |

### Output Format (JSONL)

Each line is a JSON object:

```json
{
  "row_num": 1,
  "problem_id": 1065658,
  "image_name": "04255753-0b36-495d-87eb-1e3cd0c97634.jpeg",
  "local_image": "1.jpeg",
  "questions": ["Did students label the number line correctly?", "..."],
  "ground_truth_answers": ["YES", "YES", "NO"],
  "model_answers": ["YES", "YES", "NO"],
  "comparisons": [true, true, true],
  "raw_response": "[\"YES\", \"YES\", \"NO\"]",
  "parse_ok": true,
  "elapsed_s": 2.34
}
```

> **Resuming interrupted runs:** If the output file already exists, the script checks which row numbers are present and skips them. You can safely re-run after a crash.

### Pre-computed Batches

The repository already contains several evaluation batches. Use these to skip Stage 1 for covered rows:

| File | Rows |
|------|------|
| `LLM_evaluations_QA_Teacher_0001_0100.jsonl` | 1–100 |
| `LLM_evaluations_QA_Teacher_0101_0200.jsonl` | 101–200 |
| `LLM_evaluations_QA_Teacher_0201_0500.jsonl` | 201–500 |
| `LLM_evaluations_QA_Teacher_0501_1000.jsonl` | 501–1000 |

---

## Stage 2 — Evaluate Accuracy

`scripts/evaluate_LLM_accuracy.py` loads one or more JSONL files, compares each model answer against the teacher ground truth in `DrawEduMath_SimplifiedQA.csv`, and writes a CSV accuracy report.

### Basic Usage

```bash
# Use the default output file from Stage 1
python scripts/evaluate_LLM_accuracy.py

# Specify one or more JSONL input files
python scripts/evaluate_LLM_accuracy.py \
  --input dataset/DrawEduMath/LLM_evaluations_QA_Teacher_0001_0100.jsonl \
          dataset/DrawEduMath/LLM_evaluations_QA_Teacher_0101_0200.jsonl \
  --output results/accuracy_0001_0200.csv
```

### All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH [PATH ...]` | `dataset/DrawEduMath/LLM_evaluations_QA_Teacher.jsonl` | One or more JSONL input files |
| `--output PATH` | `scripts/LLM_EVALUATION_REPORT.csv` | Output CSV report |

### Output CSV Format

The CSV has two sections:

**Summary (top rows):**

| Field | Example |
|-------|---------|
| Total tests | 400 |
| Correct | 372 |
| Accuracy | 0.930 |
| Skipped rows | 3 |

**Detail (remaining rows):**

| Column | Description |
|--------|-------------|
| `row_num` | Dataset row number |
| `problem_id` | Problem identifier |
| `image_name` | Original image filename |
| `question_idx` | 0-indexed question within the rubric |
| `question` | The rubric question text |
| `model_answer` | Gemini's answer (YES/NO/UNCLEAR) |
| `ground_truth` | Teacher's answer (YES/NO) |
| `correct` | `True`/`False` |

---

## Stage 3 — Analyze Results

`scripts/analyze_evaluations.py` reads three pre-existing CSV batches (rows 0001–0500) and prints a detailed statistical breakdown to the console.

```bash
python scripts/analyze_evaluations.py
```

> **Note:** This script has hardcoded input paths. It reads:
> - `dataset/DrawEduMath/LLM_evaluations_QA_Teacher_0001_0100.csv`
> - `dataset/DrawEduMath/LLM_evaluations_QA_Teacher_0101_0200.csv`
> - `dataset/DrawEduMath/LLM_evaluations_QA_Teacher_0201_0500.csv`
>
> To analyze different batches, edit the `INPUT_FILES` list at the top of the script.

### Output Sections

The script prints 10 analysis sections:

1. **Overall summary** — total tests, correct, incorrect, unique images and problems
2. **Per-batch accuracy** — accuracy broken down by input file
3. **Per-problem accuracy** — accuracy per problem ID
4. **Row-level accuracy distribution** — histogram of per-image scores (0–20 %, 21–40 %, …)
5. **Error analysis** — false positives, false negatives, empty/unparseable answers
6. **Question-index accuracy** — accuracy per rubric question position (Q0, Q1, Q2, …)
7. **Hardest questions** — lowest-accuracy questions (minimum 5 occurrences)
8. **Ground-truth distribution** — YES vs NO ratio in the teacher labels
9. **Confusion matrix** — precision, recall, F1 for YES/NO classification
10. **Per-question deep dive** — problem × question accuracy breakdown

---

## Recommended Workflow

```bash
# 1. Dry-run to verify setup
python scripts/generate_LLM_evaluations.py --dry-run --start 1 --end 1

# 2. Generate a small batch to check quality
python scripts/generate_LLM_evaluations.py --start 1 --end 50 \
  --output dataset/DrawEduMath/LLM_evaluations_test_batch.jsonl

# 3. Check accuracy on that batch
python scripts/evaluate_LLM_accuracy.py \
  --input dataset/DrawEduMath/LLM_evaluations_test_batch.jsonl \
  --output scripts/test_batch_accuracy.csv

# 4. Scale up
python scripts/generate_LLM_evaluations.py --start 1 --end 2030 --workers 8

# 5. Full accuracy report
python scripts/evaluate_LLM_accuracy.py

# 6. Deep-dive analysis (on existing 0001–0500 batches)
python scripts/analyze_evaluations.py
```

---

## Troubleshooting

**`FileNotFoundError` for local images**  
The script expects images in `dataset/DrawEduMath/Claude_Postprocessing/` named `{row_number}.{jpg,png}`. If images are missing for a row, that row is skipped and recorded as an error in the JSONL.

**`parse_ok: false` in JSONL output**  
Gemini returned a response that couldn't be parsed as a JSON array of YES/NO strings. This row still appears in the JSONL but is counted as "skipped" in accuracy reports. It may help to retry those rows; the script skips already-processed rows so you can append to the same output file.

**API quota / rate limit errors**  
Reduce `--workers` (try `--workers 1` or `--workers 2`) or add delays by processing smaller ranges across multiple runs.

**`GEMINI_API_KEY` not found**  
The `.env` file must be at the repository root (the same directory as `scripts/`). See `INSTALL.md` for details.

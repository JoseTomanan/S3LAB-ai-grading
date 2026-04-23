"""
Analyze LLM evaluation reports across the three QA Teacher CSV files.

Reads:
  dataset/DrawEduMath/LLM_evaluations_QA_Teacher_0001_0100.csv
  dataset/DrawEduMath/LLM_evaluations_QA_Teacher_0101_0200.csv
  dataset/DrawEduMath/LLM_evaluations_QA_Teacher_0201_0500.csv

Each CSV has a 3-row summary header followed by a blank row, then column headers,
then the data rows.  This script skips those header rows and loads only the data.

Usage:
    python scripts/analyze_evaluations.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "dataset" / "DrawEduMath"

CSV_FILES = {
    "0001-0100": DATA_DIR / "LLM_evaluations_QA_Teacher_0001_0100.csv",
    "0101-0200": DATA_DIR / "LLM_evaluations_QA_Teacher_0101_0200.csv",
    "0201-0500": DATA_DIR / "LLM_evaluations_QA_Teacher_0201_0500.csv",
}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv(path: Path, label: str) -> pd.DataFrame:
    """Load one evaluation CSV, skipping the 4-row preamble."""
    df = pd.read_csv(path, skiprows=5)
    df["batch"] = label
    df["correct"] = df["correct"].map({"True": True, "False": False}).fillna(df["correct"])
    df["correct"] = df["correct"].astype(bool)
    return df


def pct(n: int, d: int) -> str:
    return f"{n/d*100:.1f}%" if d else "—"


def divider(char: str = "─", width: int = 60) -> None:
    print(char * width)


def section(title: str) -> None:
    print()
    divider("═")
    print(f"  {title}")
    divider("═")


def subsection(title: str) -> None:
    print()
    print(f"  {title}")
    divider()


# ── load ─────────────────────────────────────────────────────────────────────

frames = []
for label, path in CSV_FILES.items():
    if not path.exists():
        print(f"WARNING: file not found — {path}")
        continue
    frames.append(load_csv(path, label))

df = pd.concat(frames, ignore_index=True)

# ── 1. Overall summary ────────────────────────────────────────────────────────

section("1. OVERALL SUMMARY")

total   = len(df)
correct = df["correct"].sum()
print(f"  Total questions evaluated : {total}")
print(f"  Correct                   : {correct}  ({pct(correct, total)})")
print(f"  Incorrect                 : {total - correct}  ({pct(total - correct, total)})")
print(f"  Unique images             : {df['image_name'].nunique()}")
print(f"  Unique problems (IDs)     : {df['problem_id'].nunique()}")
print(f"  Unique row numbers        : {df['row_num'].nunique()}")

# ── 2. Per-batch accuracy ─────────────────────────────────────────────────────

section("2. ACCURACY BY BATCH FILE")

batch_stats = (
    df.groupby("batch")["correct"]
    .agg(total="count", correct="sum")
    .assign(accuracy=lambda x: x["correct"] / x["total"] * 100)
    .sort_index()
)
batch_stats["accuracy"] = batch_stats["accuracy"].map("{:.1f}%".format)
print(batch_stats.to_string())

# ── 3. Per-problem accuracy ───────────────────────────────────────────────────

section("3. ACCURACY BY PROBLEM ID")

prob_stats = (
    df.groupby("problem_id")["correct"]
    .agg(total="count", correct="sum")
    .assign(accuracy=lambda x: x["correct"] / x["total"] * 100)
    .sort_values("accuracy")
)

print(f"\n  {'problem_id':<12} {'total':>7} {'correct':>9} {'accuracy':>10}")
divider()
for pid, row in prob_stats.iterrows():
    print(f"  {pid:<12} {int(row['total']):>7} {int(row['correct']):>9} {row['accuracy']:>9.1f}%")
print()
print(f"  Best  problem: {prob_stats['accuracy'].idxmax()}  "
      f"({prob_stats['accuracy'].max():.1f}%)")
print(f"  Worst problem: {prob_stats['accuracy'].idxmin()}  "
      f"({prob_stats['accuracy'].min():.1f}%)")

# ── 4. Per-row (image) accuracy ───────────────────────────────────────────────

section("4. ROW-LEVEL ACCURACY DISTRIBUTION")

row_acc = (
    df.groupby(["row_num", "image_name", "problem_id"])["correct"]
    .agg(total="count", correct="sum")
    .assign(accuracy=lambda x: x["correct"] / x["total"] * 100)
    .reset_index()
)

bins   = [0, 20, 40, 60, 80, 100.001]
labels = ["0–20%", "21–40%", "41–60%", "61–80%", "81–100%"]
row_acc["bucket"] = pd.cut(row_acc["accuracy"], bins=bins, labels=labels, right=False)

subsection("Distribution of per-row accuracy")
bucket_counts = row_acc["bucket"].value_counts().sort_index()
for bucket, count in bucket_counts.items():
    bar = "█" * int(count / max(bucket_counts) * 30)
    print(f"  {bucket:>8}  {count:>4}  {bar}")

subsection("Rows with all questions correct (100%)")
perfect = row_acc[row_acc["accuracy"] == 100]
print(f"  {len(perfect)} / {len(row_acc)} rows  ({pct(len(perfect), len(row_acc))})")

subsection("Rows with all questions wrong (0%)")
zero = row_acc[row_acc["accuracy"] == 0]
print(f"  {len(zero)} / {len(row_acc)} rows  ({pct(len(zero), len(row_acc))})")

# ── 5. Error analysis ─────────────────────────────────────────────────────────

section("5. ERROR ANALYSIS")

errors = df[~df["correct"]].copy()
errors["error_type"] = "other"
errors.loc[
    (errors["model_answer"].str.upper() == "YES") &
    (errors["ground_truth"].str.upper() == "NO"),
    "error_type"
] = "false_positive"
errors.loc[
    (errors["model_answer"].str.upper() == "NO") &
    (errors["ground_truth"].str.upper() == "YES"),
    "error_type"
] = "false_negative"
errors.loc[
    errors["model_answer"].fillna("").str.strip() == "",
    "error_type"
] = "empty_answer"

error_counts = errors["error_type"].value_counts()
subsection("Error type breakdown")
for etype, count in error_counts.items():
    print(f"  {etype:<20} {count:>5}  ({pct(count, len(errors))} of errors, "
          f"{pct(count, total)} of all)")

# ── 6. Question-level analysis ────────────────────────────────────────────────

section("6. QUESTION-INDEX ACCURACY  (question_idx within each image)")

qi_stats = (
    df.groupby("question_idx")["correct"]
    .agg(total="count", correct="sum")
    .assign(accuracy=lambda x: x["correct"] / x["total"] * 100)
)
print(f"\n  {'question_idx':>13} {'total':>7} {'correct':>9} {'accuracy':>10}")
divider()
for qi, row in qi_stats.iterrows():
    print(f"  {qi:>13} {int(row['total']):>7} {int(row['correct']):>9} {row['accuracy']:>9.1f}%")

# ── 7. Hardest questions ──────────────────────────────────────────────────────

section("7. HARDEST QUESTIONS  (lowest accuracy, min 5 occurrences)")

q_stats = (
    df.groupby("question")["correct"]
    .agg(total="count", correct="sum")
    .assign(accuracy=lambda x: x["correct"] / x["total"] * 100)
    .query("total >= 5")
    .sort_values("accuracy")
    .head(10)
    .reset_index()
)

for _, row in q_stats.iterrows():
    q_short = (row["question"][:70] + "…") if len(row["question"]) > 70 else row["question"]
    print(f"\n  [{row['accuracy']:.0f}%]  {q_short}")
    print(f"          correct {int(row['correct'])} / {int(row['total'])}")

# ── 8. Ground-truth answer distribution ──────────────────────────────────────

section("8. GROUND-TRUTH ANSWER DISTRIBUTION")

gt_counts = df["ground_truth"].str.upper().value_counts()
subsection("All values")
for val, cnt in gt_counts.items():
    print(f"  {val:<30} {cnt:>5}  ({pct(cnt, total)})")

# ── 9. Confusion matrix (YES/NO questions only) ───────────────────────────────

section("9. CONFUSION MATRIX  (YES/NO questions only)")

yn = df[
    df["ground_truth"].str.upper().isin(["YES", "NO"]) &
    df["model_answer"].str.upper().isin(["YES", "NO"])
].copy()

yn["gt_yn"]    = yn["ground_truth"].str.upper()
yn["model_yn"] = yn["model_answer"].str.upper()

cm = pd.crosstab(
    yn["model_yn"], yn["gt_yn"],
    rownames=["Model →"], colnames=["Ground Truth ↓"]
)
print()
print(cm.to_string())

yn_total   = len(yn)
yn_correct = (yn["gt_yn"] == yn["model_yn"]).sum()
print(f"\n  YES/NO accuracy: {yn_correct} / {yn_total}  ({pct(yn_correct, yn_total)})")

tp = len(yn[(yn["model_yn"] == "YES") & (yn["gt_yn"] == "YES")])
fp = len(yn[(yn["model_yn"] == "YES") & (yn["gt_yn"] == "NO")])
fn = len(yn[(yn["model_yn"] == "NO")  & (yn["gt_yn"] == "YES")])
tn = len(yn[(yn["model_yn"] == "NO")  & (yn["gt_yn"] == "NO")])

precision = tp / (tp + fp) if (tp + fp) else 0
recall    = tp / (tp + fn) if (tp + fn) else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print(f"  Precision  : {precision:.3f}")
print(f"  Recall     : {recall:.3f}")
print(f"  F1 score   : {f1:.3f}")

# ── 10. Per-question deep dive ────────────────────────────────────────────────

section("10. PER-QUESTION DEEP DIVE  (every unique problem × question pair)")

q_deep = (
    df.dropna(subset=["question"])
    .groupby(["problem_id", "question"])["correct"]
    .agg(total="count", correct="sum")
    .assign(accuracy=lambda x: x["correct"] / x["total"] * 100)
    .sort_values(["problem_id", "accuracy"])
    .reset_index()
)

print(f"\n  {'problem_id':<12} {'n':>3}  {'acc':>6}  question")
divider()

prev_pid = None
for _, row in q_deep.iterrows():
    pid = row["problem_id"]
    q   = str(row["question"]).strip() or "(blank)"
    q   = (q[:80] + "…") if len(q) > 80 else q
    if prev_pid is not None and pid != prev_pid:
        print()
    print(f"  {pid:<12} {int(row['total']):>3}  {pct(int(row['correct']), int(row['total'])):>6}  {q}")
    prev_pid = pid

print(f"\n  {len(q_deep)} unique (problem_id, question) pairs")

print()
divider("═")
print("  Analysis complete.")
divider("═")

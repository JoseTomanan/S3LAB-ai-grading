# Automated Testing Guide

This guide covers all automated tests in the SIPAT.MATH project: backend API tests (pytest), backend CV pipeline tests, and frontend E2E tests (Playwright).

---

## Overview

| Suite | Framework | Location | Requires API key? | Requires servers? |
|-------|-----------|----------|-------------------|-------------------|
| Backend API tests | pytest + FastAPI TestClient | `app/backend/tests/` | No | No |
| CV pipeline tests | Python scripts | `app/backend/scripts/tester/` | For `ScanLabelEvaluate.py` only | No |
| CV module tests | Python `__main__` blocks | `app/backend/logic/` | No | No |
| Frontend E2E tests | Playwright | `app/frontend/tests/` | Indirectly (backend) | Yes (both) |

---

## 1. Backend API Tests (pytest)

### What They Cover

51 tests covering all REST API endpoints. They run against an **in-memory SQLite database** — no real database, no Gemini API key, and no running server required.

Endpoints tested:

- `GET /api/health`
- `GET/POST /api/sections`, `GET /api/sections/{id}`
- `GET/POST/PATCH/DELETE /api/students`
- `GET/POST/PATCH/DELETE /api/test_instances`, `GET /api/test_instances/{id}/export`
- `GET/POST/PATCH/DELETE /api/test_items`
- `POST /api/student_answers/image_preprocess`, `/label_save_boxes`, `/commit_boxes`
- `PATCH /api/student_answers/{id}/update_answer_segmentation`
- Various 404/415/400 error paths

Image processing endpoints are tested for request validation only (unsupported file type, missing test/student records). The full OpenCV + Gemini pipeline is **not** exercised.

### Setup

```bash
cd app/backend

# Activate your virtual environment if not already active
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install dependencies (pytest is included in requirements.txt)
pip install -r requirements.txt
```

### Running the Tests

```bash
cd app/backend
python -m pytest tests/test_api.py -v
```

Useful pytest flags:

```bash
# Run a specific test by name
python -m pytest tests/test_api.py -v -k "test_health"

# Stop on first failure
python -m pytest tests/test_api.py -x

# Show local variable values on failure
python -m pytest tests/test_api.py -v --tb=long

# Quiet output (dots only)
python -m pytest tests/test_api.py -q
```

### How the Test Database Works

`tests/conftest.py` patches the database before any app code imports:

1. Sets `DATABASE_URL=sqlite://` (bare URI = fully in-memory)
2. Replaces the SQLAlchemy engine with one using `StaticPool` so the app's `get_session()` and the test fixtures share the same in-memory database
3. Each test function gets a fresh database via the `autouse=True` fixture

This means tests are fully isolated and leave no files on disk.

### Test Data

Each test starts with this pre-loaded state:

- **Sections:** "3-Rizal", "3-Aguinaldo"
- **Students:** 4 students distributed across the two sections
- **Test instances:** 2 instances with 3 test items total

---

## 2. Backend CV Pipeline Tests

These scripts test the computer vision pipeline against real images. Place your test images in `app/backend/TEMP/input/` before running.

### 2.1 Box Segmentation Test

`app/backend/scripts/tester/Proper.py` — tests answer-box detection on images whose filename starts with `proper`.

```bash
# Place images in app/backend/TEMP/input/ named proper*.{jpeg,jpg,png}
cd app/backend
python scripts/tester/Proper.py
```

**What it does:**
1. Perspective-corrects each image (removes skew from scanning angle)
2. Detects answer boxes using blob corner detection
3. Saves each detected box to `TEMP/output/{filename}/boxed/{index}.jpg`

**Output location:** `app/backend/TEMP/output/`

---

### 2.2 End-to-End Evaluation Test

`app/backend/scripts/tester/ScanLabelEvaluate.py` — full pipeline test: scan → detect boxes → label with AI → evaluate answers.

**Requires `GEMINI_API_KEY` in `.env`.**

```bash
# Place images in app/backend/TEMP/input/ named proper*.{jpeg,jpg,png}
cd app/backend
python scripts/tester/ScanLabelEvaluate.py
```

**What it does:**
1. Perspective-corrects the image
2. Detects answer boxes
3. Uses Gemini to identify which item number each box belongs to
4. Evaluates short-form answers (items 1a, 1b) against expected answers
5. Evaluates problem-solving answers (items 2, 3) against rubric questions
6. Prints YES/NO evaluation results to stdout

> The expected answers and rubric questions are hardcoded in the script and match a specific sample test paper. Use this to validate that the full CV + AI pipeline runs without errors.

---

### 2.3 CV Module Standalone Tests

`logic/box_segmenter.py` and `logic/document_scanner.py` each have a `__main__` block for quick standalone runs.

```bash
cd app/backend

# Test box segmentation (reads from TEMP/input/, writes to TEMP/output/)
python logic/box_segmenter.py

# Test document scanning only
python logic/document_scanner.py
```

Place test images in `app/backend/TEMP/input/` before running. Debug output images are written to `app/backend/TEMP/output/`.

---

## 3. Frontend E2E Tests (Playwright)

### What They Cover

4 end-to-end tests for the **Bulk Add Students** feature (`/sections/[id]`):

1. Submit button is disabled when the textarea is empty
2. A malformed line (missing comma) shows a validation error without making an API call
3. Happy path: valid CSV input adds students, closes the dialog, and updates the list
4. Partial failure: a duplicate student in the list keeps the dialog open and shows a per-line error

### Prerequisites

Both servers must be running before tests execute:

```bash
# Terminal 1 — backend
cd app/backend
uvicorn main:app --reload --port 8111

# Terminal 2 — frontend (use HTTPS, as playwright.config.ts uses https://localhost:5173)
cd app/frontend
npm run dev
```

Install Playwright browsers if this is your first run:

```bash
cd app/frontend
npx playwright install chromium
```

### Running the Tests

```bash
cd app/frontend

# Run all E2E tests
npx playwright test

# Run only the bulk-add-students spec
npx playwright test tests/bulk-add-students.spec.ts

# Run with the interactive UI
npx playwright test --ui

# Show a report after the run
npx playwright show-report
```

### Configuration

`app/frontend/playwright.config.ts`:

- **Base URL:** `https://localhost:5173`
- **Browser:** Chromium only
- **HTTPS errors ignored** (`ignoreHTTPSErrors: true`) — required because the dev server uses a self-signed certificate

### Test Data and Cleanup

The happy-path and partial-failure tests create real students via the backend API. An `afterAll` hook deletes them after the test suite finishes. If a test run is interrupted, orphan students may remain in the database — they can be removed manually through the UI or by resetting `test.db`.

---

## Quick Reference

```bash
# Backend API tests (no servers, no API key)
cd app/backend && python -m pytest tests/test_api.py -v

# CV box segmentation (requires images in TEMP/input/)
cd app/backend && python scripts/tester/Proper.py

# CV full pipeline (requires GEMINI_API_KEY + images in TEMP/input/)
cd app/backend && python scripts/tester/ScanLabelEvaluate.py

# Frontend E2E (requires backend on :8111 and frontend on :5173)
cd app/frontend && npx playwright test
```

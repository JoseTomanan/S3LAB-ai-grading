# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI backend for an AI-powered student assessment grading system. It processes scanned images of handwritten student answers, segments them into individual answer boxes using computer vision, then evaluates answers against rubrics using the Google Gemini API.

## Commands

### Run the server
```bash
cd app/backend
uvicorn main:app --reload
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run CV module standalone tests
The `logic/*.py` files with `if __name__ == "__main__"` blocks and scripts in `scripts/tester/` are run directly:
```bash
python logic/box_segmenter.py    # Tests box segmentation pipeline
python logic/document_scanner.py # Tests document scanning
python scripts/tester/Proper.py  # Tests on current dataset
```
These use images from `TEMP/input/` and output debug images to `TEMP/output/`.

## Architecture

### Processing Pipeline
The core workflow is: **upload image -> scan/deskew page -> segment answer boxes -> label boxes with AI -> evaluate answers with AI -> export scores**.

1. **DocumentScanner** (`logic/document_scanner.py`) - Finds document outline via Canny edge detection + contour detection, then perspective-warps to a flat scan. Base class for BoxSegmenter.
2. **BoxSegmenter** (`logic/box_segmenter.py`) - Extends DocumentScanner. Detects answer section boundaries using blob detection (filled dots at corners). Uses `BlobDetector` to find corner dots, groups them into quads, validates as rectangles.
3. **AIAnswerEvaluator** (`logic/ai_interface.py`) - Wraps Google Gemini API. Evaluates handwritten answers against expected answers or rubric questions. Returns YES/NO/UNCLEAR responses. Also identifies encircled item numbers on answer sheets.
4. **SheetsExporter** (`logic/sheets_exporter.py`) - Exports grading results to Excel spreadsheets via openpyxl.

### Key Design Patterns
- All images flow through the system as `bytes` (JPEG-encoded). CV classes decode/encode internally using OpenCV.
- Images are normalized to `NORMAL_SIZE` (2048px height) during processing. Size/area constants in `core/constants.py` are derived from this.
- `utils/__init__.py` contains `mapp()` (reorders 4 points to TL/TR/BR/BL order) and `is_valid_quad()` — used throughout the CV pipeline.
- The `logic/utility.py` module contains business logic functions imported by both `main.py` and routers. It instantiates singleton `AIAnswerEvaluator`, `DocumentScanner`, and `ImageModifier` objects at module level.

### Data Model
SQLModel (SQLAlchemy + Pydantic) with these entities: Section -> Student, Section -> TestInstance -> TestItem, TestInstance + Student -> TestPaperInstance -> StudentAnswer. Relationships are defined but commented out in `models.py`.

### Database
- Development: SQLite (`test.db`) — configured via `ENVIRONMENT` env var in `core/database.py`
- Production: PostgreSQL (set `ENVIRONMENT=production` and `DATABASE_URL`)
- Session management uses FastAPI dependency injection via `get_session()`

### API Structure
All routes under `/api/`. Routers in `routers/`:
- `test_instances` - CRUD for test instances
- `test_items` - CRUD for test items (questions/rubrics)
- `students` - Student management
- `student_answers` - Image upload, CV processing, box segmentation, AI evaluation, score export
- `sections` - Section management

### Scoring System
Rubric questions use semicolon-delimited format with point values in brackets: `"Did the student set up the equation? [2pts];Did they solve correctly? [3pts]"`. AI responses are semicolon-delimited YES/NO. Score calculation parses the `[Npts]` notation from rubric strings.

### Environment Variables
- `GEMINI_API_KEY` - Required for AI evaluation (Google Gemini)
- `ENVIRONMENT` - `development` (default, SQLite) or `production` (PostgreSQL)
- `DATABASE_URL` - Database connection string (optional, has defaults)

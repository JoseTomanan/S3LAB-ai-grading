# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SIPAT.MATH** — a full-stack AI-powered student assessment grading system. Scanned images of handwritten student answers are processed through a computer vision pipeline (document scanning, answer box segmentation), then evaluated against rubrics using Google Gemini AI, with scores exportable to Excel.

See `frontend/CLAUDE.md` and `backend/CLAUDE.md` for detailed per-project guidance.

## Quick Start

```bash
# Backend (FastAPI on :8000)
cd app/backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend (SvelteKit on :5173, proxies /api/* to :8000)
cd app/frontend
npm install
npm run dev
```

Both servers must be running — the frontend proxies all `/api/*` requests to the backend via Vite dev server config.

## Commands Reference

### Backend
```bash
cd app/backend
uvicorn main:app --reload          # Dev server
pytest test_api.py -v              # API tests
python logic/box_segmenter.py      # CV standalone test (uses TEMP/input/)
python logic/document_scanner.py   # Document scanner standalone test
```

### Frontend
```bash
cd app/frontend
npm run dev              # Dev server
npm run build            # Production build
npm run check            # TypeScript type checking
npm run lint             # Prettier + ESLint check
npm run format           # Auto-format
```

## Architecture

### Tech Stack
- **Frontend:** SvelteKit 2 + Svelte 5 (runes), Tailwind CSS 4, shadcn-svelte, fabric.js, TypeScript strict mode
- **Backend:** FastAPI, SQLModel (SQLAlchemy + Pydantic), OpenCV, Google Gemini API
- **Database:** SQLite (dev) / PostgreSQL (prod), controlled by `ENVIRONMENT` env var

### Core Processing Pipeline
Upload image → DocumentScanner (deskew/warp) → BoxSegmenter (detect answer boxes via blob corner detection) → AIAnswerEvaluator (Gemini API: evaluate against rubrics) → SheetsExporter (Excel export)

### Data Flow
- All images flow as JPEG-encoded `bytes` throughout the system
- Images normalized to 2048px height (`NORMAL_SIZE` in `core/constants.py`)
- Frontend uses native `fetch()` to `/api/*` endpoints — no HTTP client library
- Rubric format: semicolon-delimited questions with `[Npts]` notation (e.g., `"Did X? [2pts];Did Y? [3pts]"`)

### Data Model (SQLModel)
```
Section ──┬── Student
          └── TestInstance ── TestItem
                  │
                  └── TestPaperInstance (links TestInstance + Student)
                            └── StudentAnswer (has image, AI evaluation, scores)
```

### Key Backend Singletons (`logic/utility.py`)
`AIAnswerEvaluator`, `DocumentScanner`, `ImageModifier` — instantiated at module level and imported by routers.

### Environment Variables
- `GEMINI_API_KEY` — required for AI evaluation
- `ENVIRONMENT` — `development` (SQLite) or `production` (PostgreSQL)
- `DATABASE_URL` — optional, has defaults per environment

## Code Conventions

### Frontend
- Svelte 5 runes (`$state`, `$derived`, `$effect`) — no legacy stores
- Tabs, single quotes, trailing comma: none (see `.prettierrc`)
- shadcn-svelte components in `src/lib/components/ui/` — add via `npx shadcn-svelte@latest add <component>`
- Tailwind classes auto-sorted by Prettier

### Backend
- FastAPI dependency injection for database sessions (`get_session()`)
- Router files in `routers/`, CV/AI logic in `logic/`, models in `models.py`, schemas in `schemas.py`
- CV standalone tests use `TEMP/input/` and `TEMP/output/` directories

# Developer Setup Guide

This guide covers everything a new developer needs to get **SIPAT.MATH** running locally.

---

## Prerequisites

| Tool | Minimum Version | Notes |
|------|----------------|-------|
| Python | 3.10+ | Tested with 3.11/3.12 |
| Node.js | 20+ | LTS recommended |
| npm | 9+ | Bundled with Node.js |
| Git | any | — |

> **No Docker required.** Both services run directly on your machine.

---

## 1. Clone the Repository

```bash
git clone <repository-url>
cd S3LAB-ai-grading
```

---

## 2. Set Up Environment Variables

The application requires a Google Gemini API key. Create a `.env` file at the **repository root**:

```bash
# .env (at repo root — never commit this file)
GEMINI_API_KEY=<your_gemini_api_key_here>
```

To get a key, visit [Google AI Studio](https://aistudio.google.com/app/apikey).

The backend also reads two optional variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENVIRONMENT` | `development` | Set to `production` to use PostgreSQL instead of SQLite |
| `DATABASE_URL` | `sqlite:///./test.db` | Override the database connection string |

---

## 3. Backend Setup (FastAPI)

```bash
cd app/backend

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# Start the development server on port 8111
uvicorn main:app --reload --port 8111
```

The backend will be available at `http://localhost:8111`. The SQLite database (`test.db`) is created automatically on first run.

> **Important:** The frontend Vite proxy targets port `8111`. You must run the backend on that port for the frontend to communicate with it.

---

## 4. Frontend Setup (SvelteKit)

Open a **second terminal**:

```bash
cd app/frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:5173`. All `/api/*` requests are automatically proxied to the backend at `http://localhost:8111`.

---

## 5. Verify the Setup

With both servers running, open `http://localhost:5173` in your browser.

To confirm the backend and Gemini API are healthy:

```bash
curl http://localhost:8111/api/health
# Expected: {"status": "ok"}
# If 503: check that your GEMINI_API_KEY in .env is valid
```

---

## 6. Running Tests

### Backend API Tests

The API tests use an in-memory SQLite database and do **not** require a Gemini API key.

```bash
cd app/backend
source .venv/bin/activate
python -m pytest tests/test_api.py -v
```

### Computer Vision Standalone Tests

Place test images in `app/backend/TEMP/input/` then run the CV modules directly:

```bash
cd app/backend
python logic/box_segmenter.py      # Answer box segmentation
python logic/document_scanner.py   # Document deskew/perspective correction
```

Debug output images are written to `app/backend/TEMP/output/`.

### Frontend Type Checking and Linting

```bash
cd app/frontend
npm run check   # TypeScript type checking
npm run lint    # Prettier + ESLint
npm run format  # Auto-format files
```

---

## Project Structure

```
S3LAB-ai-grading/
├── app/
│   ├── backend/           # FastAPI backend (Python)
│   │   ├── core/          # Database engine & constants
│   │   ├── logic/         # CV pipeline & AI interface
│   │   ├── routers/       # API route handlers
│   │   ├── schemas/       # Pydantic request/response schemas
│   │   ├── models.py      # SQLModel data model
│   │   ├── main.py        # FastAPI app entry point
│   │   └── requirements.txt
│   └── frontend/          # SvelteKit frontend (TypeScript)
│       ├── src/
│       │   ├── lib/       # Shared components, types, utilities
│       │   └── routes/    # SvelteKit file-based routes
│       └── package.json
├── scripts/               # Data analysis & batch processing scripts
├── dataset/               # DrawEduMath dataset
├── crude/                 # Deprecated CLI tool (do not use)
├── .env                   # Local env vars (git-ignored, you must create this)
└── INSTALL.md             # This file
```

---

## Architecture Overview

```
Browser (SvelteKit :5173)
    │
    │  /api/* proxied by Vite
    ▼
FastAPI backend (:8111)
    │
    ├── DocumentScanner     ← perspective-corrects scanned pages
    ├── BoxSegmenter        ← detects answer boxes via blob corner detection
    ├── AIAnswerEvaluator   ← evaluates handwriting against rubrics (Gemini)
    └── SheetsExporter      ← exports scores to Excel (openpyxl)
```

**Database:** SQLite file (`test.db`) in development; PostgreSQL in production.  
**AI model:** Google Gemini 2.5 Pro / Flash via `google-genai` SDK.

---

## Common Issues

**Backend fails to start with `ModuleNotFoundError`**  
Make sure your virtual environment is activated and you ran `pip install -r requirements.txt` from inside `app/backend/`.

**Frontend shows network errors or blank data**  
The backend must be running on port `8111` before you start the frontend. Check that `uvicorn` started without errors and that `curl http://localhost:8111/api/health` responds.

**`/api/health` returns 503**  
Your `GEMINI_API_KEY` in `.env` is missing or invalid. The `.env` file must be at the repository root (the same directory that contains `app/`).

**`test.db` permission errors on Windows**  
Run the terminal as administrator or place the project in a directory where your user has write access.

---

## Further Reading

- `app/CLAUDE.md` — full project overview, data model, and code conventions  
- `app/backend/CLAUDE.md` — backend architecture and CV pipeline details  
- `app/frontend/CLAUDE.md` — frontend routing, state management, and styling  

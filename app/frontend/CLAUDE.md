# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SvelteKit frontend for SIPAT.MATH, an AI-powered student assessment grading system. Provides interfaces for managing test instances, uploading/processing student papers, reviewing AI evaluations, and manual grading. Pairs with the FastAPI backend at `../backend`.

## Commands

### Run dev server
```bash
cd app/frontend
npm run dev
```

### Build for production
```bash
npm run build
npm run preview  # preview the production build
```

### Type checking
```bash
npm run check           # one-shot
npm run check:watch     # watch mode
```

### Lint and format
```bash
npm run lint    # prettier --check + eslint
npm run format  # prettier --write
```

## Architecture

### Tech Stack
- **SvelteKit 2** with **Svelte 5** (runes: `$state`, `$derived`, `$effect`)
- **Tailwind CSS 4** with shadcn-svelte components (`src/lib/components/ui/`)
- **TypeScript** in strict mode
- **fabric.js** for canvas-based image manipulation
- Icons via unplugin-icons (Lucide + MDI icon sets)

### API Connection
- All API calls go to `/api/*` — Vite dev server proxies these to `http://localhost:8000` (the FastAPI backend)
- `VITE_API_BASE_URL` env var configures the base URL (used in `src/lib/constants.ts`)
- Uses native `fetch()` — no HTTP client library
- Data loading happens in SvelteKit `+layout.ts` / `+page.ts` `load()` functions and `onMount()` hooks

### Route Structure
```
/                           - Home page
/instances                  - List test instances
/instances/[test_id]        - Specific test instance
/instances/[test_id]/items  - Test items (questions/rubrics)
/instances/[test_id]/papers - Student papers list
/instances/[test_id]/papers/upload    - Upload papers
/instances/[test_id]/papers/[student_no]          - Student paper view
/instances/[test_id]/papers/[student_no]/manual   - Manual grading
/instances/[test_id]/papers/[student_no]/process  - Auto-grading/processing
/sections                   - Section management
/sections/[section_id]      - Specific section
```

### Key Modules
- `src/lib/types/types.ts` — Core data types (TestInstance, TestPaper, StudentAnswer, TestItem, Section)
- `src/lib/types/schemas.ts` — API response schemas
- `src/lib/utils/ai_evaluations.ts` — AI evaluation parsing/display helpers
- `src/lib/utils/image_functions.ts` — Image rotation/flip utilities using canvas API
- `src/lib/components/IrregularCropper.svelte` — Polygon cropping for answer regions
- `src/lib/components/GetAIEvaluation.svelte` — Displays AI grading results

### State Management
- Svelte context API for cross-component data sharing (e.g., `TestItemsContext`)
- Local component state via `$state` runes — no global store library

### Styling
- Tailwind utility classes with shadcn-svelte component library
- Custom button classes (`button-primary`, `button-secondary`) defined in `src/app.css`
- Theme variables in `src/lib/styles/theme.css` with dark mode support
- Prettier auto-sorts Tailwind classes via `prettier-plugin-tailwindcss`

### Code Style
- Tabs for indentation, single quotes (configured in `.prettierrc`)
- shadcn-svelte components live in `src/lib/components/ui/` — add new ones via `npx shadcn-svelte@latest add <component>`

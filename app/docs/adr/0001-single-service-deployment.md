# Single-service production deployment via adapter-static + FastAPI

The app was previously split across two free-tier deployments (SvelteKit on Vercel, FastAPI on Render), which is fragile and introduces cross-origin complexity. We decided to consolidate into a single Render web service: SvelteKit is built as static files using `adapter-static` (with a `200.html` SPA fallback), and FastAPI serves those files alongside the API. All `/api/*` requests are handled by FastAPI; everything else falls through to the SPA shell.

## Considered options

- **`adapter-node` + separate Node process** — tried previously, abandoned; running two processes on Render requires a paid plan or creative workarounds.
- **`adapter-static` served by FastAPI** — chosen. FastAPI already imports `StaticFiles`; building to static files removes the Node runtime dependency entirely in production.
- **Client-side storage (IndexedDB/SQLite WASM + OPFS)** — ruled out; conflicts with the planned login and cross-device storage features, which require server-side persistence.

## Consequences

SvelteKit SSR is not available in production. All data fetching must happen client-side (in `+page.ts` / `+layout.ts` load functions, not `+page.server.ts`). This is already the case — no server-side load functions exist.

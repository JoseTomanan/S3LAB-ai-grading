# Browser Automation

Use `npx playwright` CLI directly via Bash. Do not use Playwright MCP.

## Running tests
npx playwright test
npx playwright test --ui          # headed mode
npx playwright test path/to/spec  # single file

## Debugging a failing test
npx playwright test --debug path/to/spec
npx playwright show-report

## Codegen (record interactions)
npx playwright codegen https://localhost:5173

## Conventions
- Tests live in `tests/`
- Use `page.getByRole` and `page.getByLabel` over CSS selectors
- Prefer `waitFor` over arbitrary timeouts

## Dev server
UI runs at https://localhost:5173 (Vite). Server is assumed to already be running.
Do not attempt to start or restart it.

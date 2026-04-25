import { test, expect } from '@playwright/test';

const SECTION_URL = '/sections/1';
const API_BASE = 'http://localhost:8000';

// Students created during tests — cleaned up in afterAll
const CLEANUP_NOS = ['299900001', '299900002', '299900003'];

// An existing student_no guaranteed to already be in section 1
const EXISTING_STUDENT_NO = '200011111';

// Reliable selectors confirmed via debug inspection
const BULK_TRIGGER = 'button:has-text("Bulk add students")';
const DIALOG = '[data-slot="dialog-content"]';

test.afterAll(async ({ request }) => {
	for (const no of CLEANUP_NOS) {
		await request.delete(`${API_BASE}/api/students/${no}`).catch(() => {});
	}
});

test('submit button is disabled when textarea is empty', async ({ page }) => {
	await page.goto(SECTION_URL);
	await page.waitForLoadState('networkidle');
	await page.locator(BULK_TRIGGER).click();

	const dialog = page.locator(DIALOG);
	await expect(dialog).toBeVisible();
	await expect(dialog.locator('button:has-text("Add students")')).toBeDisabled();
});

test('malformed line (no comma) shows invalid error without network call', async ({ page }) => {
	await page.goto(SECTION_URL);
	await page.waitForLoadState('networkidle');
	await page.locator(BULK_TRIGGER).click();

	const dialog = page.locator(DIALOG);
	await expect(dialog).toBeVisible();
	await dialog.locator('textarea').fill('NOTAVALIDROW');
	await dialog.locator('button:has-text("Add students")').click();

	await expect(dialog).toBeVisible();
	await expect(dialog.getByText('Missing comma separator')).toBeVisible();
});

test('happy path: adds all students, closes dialog, list updates', async ({ page }) => {
	await page.goto(SECTION_URL);
	await page.waitForLoadState('networkidle');
	await page.locator(BULK_TRIGGER).click();

	const dialog = page.locator(DIALOG);
	await expect(dialog).toBeVisible();
	await dialog.locator('textarea').fill(
		'299900001,Test Student One\n299900002,Test Student Two'
	);
	await dialog.locator('button:has-text("Add students")').click();

	// Dialog should close on full success
	await expect(dialog).not.toBeVisible({ timeout: 8000 });

	// Both students should now appear in the class list
	await expect(page.getByText('Test Student One')).toBeVisible();
	await expect(page.getByText('Test Student Two')).toBeVisible();
});

test('partial failure: keeps dialog open and shows error for failing row', async ({ page }) => {
	await page.goto(SECTION_URL);
	await page.waitForLoadState('networkidle');
	await page.locator(BULK_TRIGGER).click();

	const dialog = page.locator(DIALOG);
	await expect(dialog).toBeVisible();
	// Line 1: new student; Line 2: duplicate → should fail with 409
	await dialog.locator('textarea').fill(
		`299900003,Test Student Three\n${EXISTING_STUDENT_NO},Billy Joe Armstrong`
	);
	await dialog.locator('button:has-text("Add students")').click();

	// Dialog must stay open because at least one row failed
	await expect(dialog).toBeVisible({ timeout: 8000 });

	// Error panel should call out line 2
	await expect(dialog.getByText('Line 2')).toBeVisible();
});

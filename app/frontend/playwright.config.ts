import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
	use: {
		baseURL: 'https://localhost:5173',
		ignoreHTTPSErrors: true,
	},
	projects: [
		{
			name: 'chromium',
			use: { ...devices['Desktop Chrome'] },
		},
	],
});

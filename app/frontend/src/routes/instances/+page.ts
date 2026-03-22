import type { PageLoad } from './$types.ts';
import type { TestInstance } from '$lib/index.ts';

export const load: PageLoad = async ({ fetch }) => {
	const response = await fetch('/api/test_instances');

	let instances: TestInstance[] = [];
	if (response.ok) {
		const result = await response.json();
		instances = result ?? [];
	}

	return { instances };
};

import { API_URL } from '$lib/constants.ts';
import type { PageLoad } from './$types.ts';
import type { Section } from '$lib/index.ts';

export const load: PageLoad = async ({ fetch }) => {
	const response = await fetch(`${API_URL}/api/sections/`);

	let sections: Section[] = [];
	if (response.ok) {
		const result = await response.json();
		sections = result ?? [];
	}

	return { sections };
};

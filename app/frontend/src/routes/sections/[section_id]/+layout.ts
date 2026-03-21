import type { LayoutLoad } from './$types.d.ts';
import type { Student } from '$lib/index.ts';

export const load: LayoutLoad = async ({ fetch, params }) => {
	const { section_id } = params;

	let students: Student[] = [];
	const response = await fetch(`/api/sections/${section_id}`);
	if (response.ok) {
		students = await response.json();
	}

	return { section_id, students };
};
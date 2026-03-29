import { API_URL } from '$lib/constants.ts';
import type { LayoutLoad } from './$types.d.ts';
import type { Student } from '$lib/index.ts';
import { api } from '$lib/utils/api.ts';

export const load: LayoutLoad = async ({ fetch, params }) => {
	const { section_id } = params;
	const students = (await api<Student[]>(`${API_URL}/api/sections/${section_id}`, undefined, fetch)) ?? [];
	return { section_id, students };
};

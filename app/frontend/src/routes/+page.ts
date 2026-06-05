import { API_URL } from '$lib/constants.ts';
import type { PageLoad } from './$types.ts';
import type { TestInstance } from '$lib/index.ts';
import { api } from '$lib/utils/api.ts';

export const load: PageLoad = async ({ fetch }) => {
	const instances =
		(await api<TestInstance[]>(`${API_URL}/api/test_instances`, undefined, fetch)) ?? [];
	const recent = [...instances]
		.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
		.slice(0, 2);
	return { recent };
};

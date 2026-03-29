import { API_URL } from '$lib/constants.ts';
import type { PageLoad } from './$types.ts';
import type { GetEvaluationsResponse } from '$lib/index.ts';
import { api } from '$lib/utils/api.ts';

export const load: PageLoad = async ({ fetch, parent }) => {
	const { test_id } = await parent();

	const result = await api<{ statuses: GetEvaluationsResponse[] }>(
		`${API_URL}/api/student_answers/${test_id}/statuses`,
		undefined,
		fetch
	);
	const statuses = result?.statuses ?? [];

	return { statuses };
};

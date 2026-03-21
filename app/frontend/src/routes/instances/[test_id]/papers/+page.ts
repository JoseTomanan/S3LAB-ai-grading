import type { PageLoad } from './$types.ts';
import type { GetEvaluationsResponse } from '$lib/index.ts';

export const load: PageLoad = async ({ fetch, parent }) => {
	const { test_id } = await parent();

	let statuses: GetEvaluationsResponse[] = [];
	const response = await fetch(`/api/student_answers/${test_id}/statuses`);
	if (response.ok) {
		const result = await response.json();
		statuses = result.statuses ?? [];
	}

	return { statuses };
};

import type { LayoutLoad } from './$types.ts';

export const load: LayoutLoad = async ({ params }) => {
	return { test_id: params.test_id };
};
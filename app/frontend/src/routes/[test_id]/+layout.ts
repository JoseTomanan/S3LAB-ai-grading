import type { LayoutLoad } from './$types.d.ts';

export const load: LayoutLoad = async ({ params }) => {
	return { test_id: params.test_id };
};
import type { LayoutLoad } from './$types.d.ts';

export const load: LayoutLoad = async ({ params }) => {
	return { testid: params.testid };
};
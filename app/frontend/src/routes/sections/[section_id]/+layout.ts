import type { LayoutLoad } from './$types.d.ts';

export const load: LayoutLoad = async ({ params }) => {
	return { section_id: params.section_id };
};
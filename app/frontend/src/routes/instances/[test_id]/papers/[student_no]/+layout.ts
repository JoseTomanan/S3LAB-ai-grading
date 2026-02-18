import type { LayoutLoad } from './$types.ts';

export const load: LayoutLoad = async ({ params }) => {
	return { student_no: params.student_no };
};
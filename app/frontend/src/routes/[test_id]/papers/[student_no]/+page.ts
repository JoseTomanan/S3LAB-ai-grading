import type { PageLoad } from './$types.d.ts';

export const load: PageLoad = async ({ params }) => {
	return { student_no: params.student_no };
};
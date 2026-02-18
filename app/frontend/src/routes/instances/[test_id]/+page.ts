import { redirect } from '@sveltejs/kit';
import type { PageLoad } from './$types.ts';

export const load: PageLoad = ({ params }) => {
	// This triggers as soon as the user hits /[test_id]
	throw redirect(308, `/${params.test_id}/items`);
};
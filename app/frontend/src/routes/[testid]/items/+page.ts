
import type { PageLoad } from './$types.d.ts';

export const load: PageLoad = ({ params }) => {
	return {
		testid: params.testid
	};
};
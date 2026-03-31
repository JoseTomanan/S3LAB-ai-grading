import { API_URL } from '$lib/constants.ts';
import type { LayoutLoad } from './$types.ts';
import type { Section } from '$lib/index.ts';
import { api } from '$lib/utils/api.ts';

export const load: LayoutLoad = async ({ fetch }) => {
	const sections = (await api<Section[]>(`${API_URL}/api/sections/`, undefined, fetch)) ?? [];
	return { sections };
};

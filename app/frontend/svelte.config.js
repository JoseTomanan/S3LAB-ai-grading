import adapter from '@sveltejs/adapter-auto';
/* adapter-node removed because back to Vercel */
// import adapter from '@sveltejs/adapter-node';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: vitePreprocess(),
	kit: {
		alias: {
			$lib: 'src/lib',
      '@': 'src',
		},
		adapter: adapter()
	}
};

export default config;

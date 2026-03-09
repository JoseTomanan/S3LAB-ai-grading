import tailwindcss from '@tailwindcss/vite';
import Icons from 'unplugin-icons/vite';
import basicSsl from '@vitejs/plugin-basic-ssl'
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		tailwindcss(),
		sveltekit(),
    basicSsl(),
		Icons({
			compiler: 'svelte',
		}),
	]
});

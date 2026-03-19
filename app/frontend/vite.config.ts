import tailwindcss from '@tailwindcss/vite';
import Icons from 'unplugin-icons/vite';
import basicSsl from '@vitejs/plugin-basic-ssl'
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    // FIXME: allow host when done
    // host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },
	plugins: [
		tailwindcss(),
		sveltekit(),
    // basicSsl(),
		Icons({
			compiler: 'svelte',
		}),
	]
});

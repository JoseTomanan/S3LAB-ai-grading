import tailwindcss from '@tailwindcss/vite';
import Icons from 'unplugin-icons/vite';
import basicSsl from '@vitejs/plugin-basic-ssl';
import { sveltekit } from '@sveltejs/kit/vite';
import { enhancedImages } from '@sveltejs/enhanced-img';
import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },
  preview: {
    allowedHosts: [
      's3lab-frontend.onrender.com',
    ],
  },
	plugins: [
		tailwindcss(),
    enhancedImages(),
		sveltekit(),
    basicSsl(),
		Icons({
			compiler: 'svelte',
		}),
	]
});

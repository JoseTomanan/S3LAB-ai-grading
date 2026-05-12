import tailwindcss from '@tailwindcss/vite';
import Icons from 'unplugin-icons/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { enhancedImages } from '@sveltejs/enhanced-img';
import { defineConfig } from 'vite';
import { marked } from 'marked';

export default defineConfig({
  server: {
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8111',
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
		Icons({
			compiler: 'svelte',
		}),
    {
      name: 'md-loader',
      transform(src, id) {
        if (id.endsWith('.md')) {
          const html = marked.parse(src);
          return `export default ${JSON.stringify(html)}`;
        }
      }
    },
	]
});

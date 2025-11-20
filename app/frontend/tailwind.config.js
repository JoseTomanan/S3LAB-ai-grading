/** @type {import('tailwindcss').Config} */

export default {
	darkMode: ["class"],
	safelist: ["dark"],
	content: ["./src/**/*.{html,js,svelte,ts}"],
	theme: {
		extend: {
			fontFamily: {
			heading: "var(--font-heading)",
			sans: "var(--font-sans)",
			mono: "var(--font-mono)",
			},
			container: {
				center: true,
				padding: "2rem",
			},
		},
		screens: {
			xs: "475px",
			sm: "640px",
			md: "768px",
			lg: "1024px",
			xl: "1280px",
			xxl: "1536px",
		},
	}
}
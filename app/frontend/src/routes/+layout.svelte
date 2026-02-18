<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.ico';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>Grading AI</title>
</svelte:head>

{#if children}
	{@render children()}
{:else}
	<h2>404 PAGE NOT FOUND</h2>
	<p>If you are attempting to put a custom link, please start <a href="/">here</a>.</p>
{/if}

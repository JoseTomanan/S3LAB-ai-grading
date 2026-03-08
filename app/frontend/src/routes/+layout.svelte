<script lang="ts">
  import '../app.css';
  import favicon from '$lib/assets/favicon.ico';
	import { navigating } from '$app/state';
	import Spinner from '$lib/components/ui/spinner/spinner.svelte';

  let { children } = $props();
</script>

<svelte:head>
  <link rel="icon" href={favicon} />
  <title>LLAMATH</title>
</svelte:head>

<div class="flex justify-center bg-fixed">
  <div class="big-container">
    {@render children()}
  </div>
  <footer class="fixed bottom-0 left-0 w-full flex items-start z-50">
    {#if navigating.to}
      <Spinner class="size-8 text-chart-3 p-0.5"/>
    {/if}
  </footer>
</div>

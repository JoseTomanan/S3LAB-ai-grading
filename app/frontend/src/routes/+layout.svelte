<script lang="ts">
  import '../app.css';
  import favicon from '$lib/assets/favicon.ico';
	import { navigating } from '$app/state';
	import Spinner from '$lib/components/ui/spinner/spinner.svelte';

  let { children } = $props();
</script>


<svelte:head>
  <link rel="icon" href={favicon} />
  <title>SIPAT.MATH</title>
</svelte:head>


<div class="flex justify-center bg-fixed">
  <div class="app">
    {@render children()}
  </div>
  <span class="fixed bottom-0 left-0 w-full flex space-x-2 items-center justify-end z-50 font-semibold p-2">
    {#if navigating.to}
      <Spinner class="size-12 text-accent"/>
    {/if}
  </span>
</div>

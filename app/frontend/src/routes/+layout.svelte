<script lang="ts">
  import '../app.css';
  import favicon from '$lib/assets/favicon.ico';
	import IconAPIStatus from '~icons/mdi/circle-medium';
	import { navigating, page } from '$app/state';
	import * as Tooltip from '$lib/components/ui/tooltip/index.ts';
	import Spinner from '$lib/components/ui/spinner/spinner.svelte';
	import { Toaster } from 'svelte-5-french-toast';

  let { children } = $props();

  const isRouteHome = $derived(page.route.id! === "/");
  
  // Placeholders; TODO: Replace with actual health check logic
  let isOnline = $state(false);
  let isHealthCheckOngoing = $state(true);
  
  // Connection status simulation
  // Simulate connection with 300ms delay
  // TODO: Replace with actual health check logic
  $effect(() => {
    const timer = setTimeout(() => {
      isHealthCheckOngoing = false;
      isOnline = true;
    }, 2000);
    
    return () => clearTimeout(timer);
  });
</script>


<Toaster/>
<svelte:head>
  <link rel="icon" href={favicon} />
  <title>SIPAT.MATH</title>
</svelte:head>


<Tooltip.Provider>
  <div class="flex justify-center bg-fixed">
    <div class="app">
      {@render children()}
    </div>
    
    <div class="fixed bottom-1.75 right-2 z-50">
      {#if navigating.to}
        <Spinner class="size-12 text-accent"/>
      {/if}
    </div>

    <div class="fixed bottom-1.75 right-2 z-51 inline-flex gap-x-0.5 items-center">
      {#if isRouteHome || !isOnline}
        <span class="text-xs opacity-40">
          {isOnline
            ? 'Online'
            : (isHealthCheckOngoing
              ? 'Connecting to backend...'
              : 'Offline')}
        </span>
      {/if}
      <!-- TODO: Add non-hardcoded values -->
      <IconAPIStatus class="opacity-80
                {isOnline ? '*:text-green-600' : '*:text-gray-600'}"/>
    </div>
  </div>
</Tooltip.Provider>

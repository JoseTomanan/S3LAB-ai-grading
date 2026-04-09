<script lang="ts">
  import '../app.css';
  import favicon from '$lib/assets/favicon.ico';
	import IconAPIStatus from '~icons/mdi/circle-medium';
	import { navigating, page } from '$app/state';
	import * as Tooltip from '$lib/components/ui/tooltip/index.ts';
	import Spinner from '$lib/components/ui/spinner/spinner.svelte';
	import { getApiStatus, type ApiStatus } from '$lib/utils/api.ts';
	import { Toaster } from 'svelte-5-french-toast';

  let { children } = $props();

  const isRouteHome = $derived(page.route.id! === "/");

  let apiStatus = $state<ApiStatus | 'checking'>('checking');

  $effect.pre(() => {
    let cancelled = false;

    (async () => {
      const status = await getApiStatus();
      if (!cancelled) apiStatus = status;
    })();

    return () => { cancelled = true };
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
    
    <div class="fixed bottom-1.75 right-2 z-1">
      {#if navigating.to}
        <Spinner class="size-12 text-accent"/>
      {/if}
    </div>

    <div class="fixed bottom-1.75 right-2 z-2 inline-flex gap-x-0.5 items-center">
      {#if isRouteHome || apiStatus !== 'online'}
        <span class="text-xs opacity-40">
          {#if apiStatus === 'checking'}
            Connecting...
          {:else if apiStatus === 'online'}
            Online
          {:else if apiStatus === 'no-ai-key'}
            No AI key
          {:else}
            Offline
          {/if}
        </span>
      {/if}
      <IconAPIStatus class="opacity-80
                {apiStatus === 'online'
                  ? '*:text-green-600'
                  : apiStatus === 'checking'
                    ? '*:text-gray-600'
                    : '*:text-destructive'}
              "/>
    </div>
  </div>
</Tooltip.Provider>

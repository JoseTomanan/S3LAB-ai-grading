<script lang="ts">
  import '../app.css';
  import favicon from '$lib/assets/favicon.ico';
  import IconAPIStatus from '~icons/mdi/circle-medium';
  import { navigating } from '$app/state';
  import * as Tooltip from '$lib/components/ui/tooltip/index.ts';
  import Spinner from '$lib/components/ui/spinner/spinner.svelte';
  import { getApiStatus, type ApiStatus } from '$lib/utils/api.ts';
  import { Toaster } from 'svelte-5-french-toast';
  import BottomTabBar from '$lib/components/BottomTabBar.svelte';

  let { children } = $props();

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

<Toaster />
<svelte:head>
  <link rel="icon" href={favicon} />
  <title>SIPAT.MATH</title>
</svelte:head>

<Tooltip.Provider>
  <div class="min-h-dvh flex justify-center bg-background">
    <div class="w-full max-w-[480px] flex flex-col min-h-dvh">
      <div class="flex-1 flex flex-col min-h-0 overflow-y-auto pb-[60px]">
        {@render children()}
      </div>
      <div class="fixed bottom-0 left-0 right-0 flex justify-center z-10 pointer-events-none">
        <div class="w-full max-w-[480px] pointer-events-auto">
          <BottomTabBar />
        </div>
      </div>
    </div>
  </div>

  <!-- navigating indicator -->
  {#if navigating.to}
    <div class="fixed bottom-[68px] right-3 z-20">
      <Spinner class="size-8 text-primary" />
    </div>
  {/if}

  <!-- offline/error indicator: only shown when not online -->
  {#if apiStatus !== 'online' && apiStatus !== 'checking'}
    <div class="fixed top-3 right-3 z-20 inline-flex gap-x-1 items-center bg-card border border-border rounded-full px-2.5 py-1 shadow-sm">
      <span class="text-xs text-muted-foreground">
        {apiStatus === 'no-ai-key' ? 'No AI key' : 'Offline'}
      </span>
      <IconAPIStatus class="text-destructive" />
    </div>
  {/if}
</Tooltip.Provider>

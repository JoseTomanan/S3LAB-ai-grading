<script lang="ts">
  import { page } from '$app/state';
  import type { LayoutData } from './$types.d.ts';
  import type { Snippet } from 'svelte';
  import { setContext, untrack } from 'svelte';

  import BulkUpload from './BulkUpload.svelte';
  import ExportSheets from './ExportSheets.svelte';
  import TopBar from '$lib/components/TopBar.svelte';

  import IconUpload from '~icons/mdi/tray-upload';

  import type { TestInstance, TestItemsContext } from '$lib/index.ts';
  import { invalidateAll } from '$app/navigation';

  let { data, children }: { data: LayoutData; children: Snippet } = $props();

  const isRouteItems = $derived(page.route.id!.includes('/items'));
  const isRoutePapers = $derived(page.route.id!.includes('/papers'));

  let activeTestInstance: TestInstance = $derived(data.test_instance!);
  let isBulkUploadOpen = $state(false);

  let testItemsContext: TestItemsContext = $state({
    items: untrack(() => data.test_items!),
    isLoading: false,
  });
  $effect(() => { testItemsContext.items = data.test_items!; });
  setContext('testItemsContext', testItemsContext);
</script>

<div class="flex flex-col min-h-full">
  {#snippet rightSlot()}
    <button
      onclick={() => { isBulkUploadOpen = true; }}
      class="size-9 rounded-full bg-background border border-border flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-border transition-colors"
      title="Bulk upload"
    >
      <IconUpload class="size-[18px]" />
    </button>
    <ExportSheets test_id={data.test_id} />
  {/snippet}

  <TopBar
    title={activeTestInstance.name}
    sub={activeTestInstance.test_id}
    back="/instances"
    right={rightSlot}
  />

  <!-- tab pills + meta sub-bar -->
  <div class="bg-card border-b border-border px-3.5 pt-2.5 pb-2">
    <div class="flex bg-background rounded-md p-[3px] gap-[2px]">
      <a
        href="/instances/{data.test_id}/items"
        class="flex-1 text-center text-[13px] font-semibold py-1.5 rounded transition-all no-underline
          {isRouteItems
            ? 'bg-card text-foreground shadow-sm'
            : 'text-muted-foreground hover:text-foreground'}"
      >Items</a>
      <a
        href="/instances/{data.test_id}/papers"
        class="flex-1 text-center text-[13px] font-semibold py-1.5 rounded transition-all no-underline
          {isRoutePapers
            ? 'bg-card text-foreground shadow-sm'
            : 'text-muted-foreground hover:text-foreground'}"
      >Papers</a>
    </div>

    <div class="flex justify-between mt-2 mb-0.5">
      <span class="text-[11px] text-foreground/55">
        Created {activeTestInstance.date ? new Date(activeTestInstance.date).toLocaleDateString() : '—'}
      </span>
      <span class="text-[11px] text-foreground/55 flex items-center gap-1">
        Rendered
        {#if activeTestInstance.is_done_rendering}
          <span class="text-secondary-700">✓</span>
        {:else}
          <span class="text-muted-foreground">✗</span>
        {/if}
      </span>
    </div>
  </div>

  <div class="flex-1">
    {@render children()}
  </div>
</div>

<BulkUpload
  test_instance={activeTestInstance}
  defaultNumBoxes={data.test_items?.length ?? 2}
  onuploadcomplete={invalidateAll}
  open={isBulkUploadOpen}
/>

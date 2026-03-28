<script lang="ts">
  let { data, children }: {data: LayoutData, children: Snippet} = $props();

  import { navigating, page } from '$app/state';
  import type { LayoutData } from './$types.d.ts';
  import type { Snippet } from 'svelte';
  import { setContext } from 'svelte';

  import ExportSheets from './ExportSheets.svelte';

  import IconBack from '~icons/mdi/arrow-back';
  import IconTable from '~icons/mdi/table';
  import IconUpload from '~icons/mdi/tray-upload';
  import IconHome from '~icons/mdi/home-outline';

  import type { TestInstance, TestItemsContext } from '$lib/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.ts';
	import { Separator } from '$lib/components/ui/separator/index.ts';
	import { Skeleton } from '$lib/components/ui/skeleton/index.ts';
	import { goto } from '$app/navigation';

  if (!data.test_instance)
    throw new Error("Test instance not loaded.");
  if (!data.test_items)
    throw new Error("Test items not loaded.");

  const isRouteItems = $derived(page.route.id!.includes('/items'));
  const isRoutePapers = $derived(page.route.id!.includes('/papers'));
  const isRouteUpload = $derived(page.route.id!.includes('/upload'));

  let activeTestInstance: TestInstance = $derived(data.test_instance!);

  let testItemsContext: TestItemsContext = $state({items: data.test_items!, isLoading: false});
  $effect(() => {
    testItemsContext.items = data.test_items!;
  });
  setContext("testItemsContext", testItemsContext);
</script>



<!-- <div> -->
  <nav class="bg-sidebar text-sidebar-foreground p-4 pt-6 shadow shadow-sidebar-border space-y-2.5">
    <span class="flex flex-row items-center justify-between">
      <a href="/instances">
        <IconHome class="size-8 opacity-85" />
      </a>
      <h1>{ activeTestInstance.name }</h1>
      <button class="p-0 size-8 cursor-pointer"
              onclick={() => goto("..")}>
        <IconBack class="size-full" />
      </button>
    </span>
    <Separator/>
    <div class="flex flex-col sm:flex-row gap-x-4 gap-y-1
              *:flex *:justify-between
              [&>*>h4]:font-normal [&>*>h4]:text-foreground/85 [&>*>h4]:leading-none [&>*>h4]:whitespace-nowrap">
      <span class="flex-3/5 tracking-tight">
        <h4>TestID: {activeTestInstance.test_id}</h4>
        <h4>
          Status:
          {activeTestInstance.is_done_rendering 
            ? "☑️" 
            : "✖️" }
        </h4>
      </span>
      <span class="flex-2/5">
        <h4>
          Date:
          {activeTestInstance.date
            ? new Date(activeTestInstance.date).toLocaleDateString()
            : "" }
        </h4>
        <h4>SectionID: { activeTestInstance.section_id }</h4>
      </span>
    </div>
    <div class="flex items-center justify-between underline-offset-3 *:space-x-1 ">
      <span>
        <a class="button-primary {isRouteItems ? "ring-2 ring-primary-foreground" : ""}"
            href="/instances/{data.test_id}/items">
          Items
        </a>
        <a class="button-primary {isRoutePapers ? "ring-2 ring-primary-foreground" : ""}"
            href="/instances/{data.test_id}/papers">
          Papers
        </a>
      </span>
      <span class="flex flex-row gap-x-1">
        <Dialog.Root>
          <Dialog.Trigger class="button-primary">
            <IconTable class="size-6" />
          </Dialog.Trigger>
          <ExportSheets test_id={data.test_id}/>
        </Dialog.Root>
        <a class="button-primary {isRouteUpload ? "ring-2 ring-primary-foreground/80" : ""}"
            href="/instances/{data.test_id}/upload">
          <IconUpload class="size-6"/>
        </a>
      </span>
    </div>
  </nav>
  <div class="container -mt-4">
    {#if navigating.to}
      <Skeleton class="grayscale-50 w-full h-64 rounded-none"/>
    {:else}
      {@render children()}
    {/if}
  </div>
<!-- </div> -->

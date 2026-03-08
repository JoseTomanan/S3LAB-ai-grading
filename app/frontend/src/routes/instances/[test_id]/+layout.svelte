<script lang="ts">
  let { data, children }: {data: LayoutData, children: Snippet} = $props();

  import { navigating, page } from '$app/state';
  import type { LayoutData } from './$types.d.ts';
  import type { Snippet } from 'svelte';
  import { onMount, setContext } from 'svelte';

  import ExportSheets from './ExportSheets.svelte';
	import BulkUpload from './BulkUpload.svelte';

  import IconBack from '~icons/mdi/arrow-back';
  import IconTable from '~icons/mdi/table';
  import IconUpload from '~icons/mdi/tray-upload';
  import IconHome from '~icons/mdi/home-outline';
  
  import type { TestInstance, TestItem, TestItemsContext } from '$lib/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.ts';
	import { Separator } from '$lib/components/ui/separator/index.ts';
	import { Skeleton } from '$lib/components/ui/skeleton/index.ts';

  if (!data.test_instance)
    throw new Error("Test instance not loaded.");
  if (!data.test_items)
    throw new Error("Test items not loaded.");

  const isRouteItems = $derived(page.route.id!.includes('/items'));
  const isRoutePapers = $derived(page.route.id!.includes('/papers'));

  let activeTestInstance: TestInstance = $state({
    name: "",
    section_id: -1,
    date: "",
    test_id: data.test_id,
    is_done_rendering: false,
  });

  let testItemsContext: TestItemsContext = $state({items: [], isLoading: false});
  setContext("testItemsContext", testItemsContext);

  onMount(async () => {
    testItemsContext.isLoading = true;
    try {
      activeTestInstance = data.test_instance!;
      testItemsContext.items.push(...data.test_items!);
    } catch (e) {
      alert("Failed to fetch test instance details:\n"+e);
    } finally {
      testItemsContext.isLoading = false;
    }
  });
</script>


<div class="space-y-7 pb-3">
  <div class="bg-sidebar text-sidebar-foreground p-4 pt-8 shadow shadow-sidebar-border space-y-3">
    <span class="flex flex-row items-center justify-between">
      <button class="p-0 size-8 cursor-pointer"
              onclick={() => history.back()}>
        <IconBack class="size-full" />
      </button>
      <h1>{ activeTestInstance.name }</h1>
      <a href="/">
        <IconHome class="size-7 opacity-85" />
      </a>
    </span>
    <Separator/>
    <div id="change-instance-details"
          class="space-y-1 *:flex *:justify-between [&>*>h4]:font-normal [&>*>h4]:opacity-85 [&>*>h4]:leading-none">
      <span>
        <h4>
          TestID: {activeTestInstance.test_id}
        </h4>
        <h4 class="flex gap-1">
          Status:
          {#if activeTestInstance.is_done_rendering}
            &check;
          {:else}
            &cross;
          {/if}
        </h4>
      </span>
      <span>
        <h4>
          Date:
          {activeTestInstance.date
            ? new Date(activeTestInstance.date).toLocaleDateString()
            : "" }
        </h4>
        <h4>
          SectionID: { activeTestInstance.section_id }
        </h4>
      </span>
    </div>
    <div id="thisOne"
          class="flex items-center justify-between *:space-x-1 [&>span>a]:font-medium [&>span>a]:underline-offset-3">
      <span>
        <a class={`button-primary ${isRouteItems ? "underline" : ""}`}
            href={`/instances/${data.test_id}/items`}>
          Items
        </a>
        <a class={`button-primary ${isRoutePapers ? "underline" : ""}`}
            href={`/instances/${data.test_id}/papers`}>
          Papers
        </a>
      </span>
      <span>
        <Dialog.Root>
          <Dialog.Trigger class="button-primary">
            <IconTable class="size-6" />
          </Dialog.Trigger>
          <ExportSheets test_id={data.test_id}/>
        </Dialog.Root>
        <Dialog.Root>
          <Dialog.Trigger class="button-primary">
            <IconUpload class="size-6"/>
          </Dialog.Trigger>
          <BulkUpload test_id={data.test_id}
                      section_id={data.test_instance!.section_id}/>
        </Dialog.Root>
      </span>
    </div>
  </div>
  <div class="px-4 pb-6">
    {#if navigating.to}
      <!--  largely untested; 
        FIXME: remove this once verified that this is actually working -->
      <Skeleton class="grayscale-50 w-full h-64 rounded-none"/>
    {:else}
      {@render children()}
    {/if}
  </div>
</div>
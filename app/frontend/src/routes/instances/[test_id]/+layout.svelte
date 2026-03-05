<script lang="ts">
  import { API_BASE_URL } from '$lib/constants.ts';

  let { data, children }: {data: LayoutData, children: Snippet} = $props();

  import type { LayoutData } from './$types.d.ts';
  import type { Snippet } from 'svelte';
  import { onMount, setContext } from 'svelte';
  import ExportSheets from './ExportSheets.svelte';

  import MdiArrowBack from '~icons/mdi/arrow-back';
  import MdiCheckboxBlankOutline from '~icons/mdi/checkbox-blank-outline';
  import MdiCheckboxMarkedOutline from '~icons/mdi/checkbox-marked-outline';
  import MdiTable from '~icons/mdi/table';
  import MdiPaperAddOutline from '~icons/mdi/paper-add-outline';
  import MdiHome from '~icons/mdi/home';
  
  import type { TestInstance, TestItem, TestItemsContext } from '$lib/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.ts';

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
      const response = await fetch(
            `${API_BASE_URL}/api/test_instances/${data.test_id}`,
            {
              method: "GET",
              headers: {'Content-Type': 'application/json',},
            }
            );

      const result = await response.json();
      activeTestInstance.name = result.name;
      activeTestInstance.section_id = result.section_id;
      activeTestInstance.date = result.date;
      activeTestInstance.is_done_rendering = result.is_done_rendering;

      const responseTwo = await fetch(
            `${API_BASE_URL}/api/test_items/${data.test_id}/items`,
            {
              method: "GET",
              headers: {'Content-Type': 'application/json',},
            }
          );

      switch (responseTwo.status) {
        case 200:
          const result = await responseTwo.json();
          testItemsContext.items.push(...result.items);
          break;
        default:
          alert(`${responseTwo.status} ${responseTwo.statusText}`);
      }
    } catch (e) {
      alert("Failed to fetch test instance details:\n"+e);
    } finally {
      testItemsContext.isLoading = false;
    }
  });

  async function bulkUpload() {
    // TODO: entire function
  }
</script>


<div class="space-y-8 pb-4">
  <div class="bg-sidebar text-sidebar-foreground px-4 py-8 border-b border-sidebar-border space-y-4">
    <span class="flex flex-row items-center justify-between">
      <button class="p-0 size-8 cursor-pointer"
              onclick={() => history.back()}>
        <MdiArrowBack class="size-full" />
      </button>
      <h1>{ activeTestInstance.name }</h1>
      <a href="/"
          class="button-outline">
        <MdiHome class="size-6" />
      </a>
    </span>
    <div id="change-instance-details">
      <span class="flex justify-between">
        <h3>
          TestID:
          <span class="font-mono text-sm">{ activeTestInstance.test_id }</span>
        </h3>
        <h3 class="flex gap-0.5">
          Status:
          {#if activeTestInstance.is_done_rendering}
            <MdiCheckboxMarkedOutline class="size-6 opacity-40"/>
          {:else}
            <MdiCheckboxBlankOutline class="size-6 opacity-40" />
          {/if}
        </h3>
      </span>
      <span class="flex justify-between">
        <h3>
          Date: {activeTestInstance.date
                  ? new Date(activeTestInstance.date).toLocaleDateString()
                  : "" }
        </h3>
        <h3>
          SectionID:
          <span class="font-mono text-sm">{ activeTestInstance.section_id }</span>
        </h3>
      </span>
    </div>
    <span id="thisOne" class="flex items-center gap-2 justify-end">
      <a class="button-primary font-medium"
          href={`/instances/${data.test_id}/items`}>
        Items
      </a>
      <a class="button-primary font-medium"
          href={`/instances/${data.test_id}/papers`}>
        Papers
      </a>
      <Dialog.Root>
        <Dialog.Trigger class="button-primary">
          <MdiTable class="size-6" />
        </Dialog.Trigger>
        <ExportSheets test_id={data.test_id}/>
      </Dialog.Root>
      <button class="button-primary opacity-50"
              onclick={() => {}}
              disabled>
        <MdiPaperAddOutline class="size-6"/>
      </button>
    </span>
  </div>
  <div class="px-4">
    {@render children()}
  </div>
</div>
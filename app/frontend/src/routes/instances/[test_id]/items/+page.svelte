<script lang="ts">
  const { data } = $props();

  import { getContext } from "svelte";
  
  import MdiEditOutline from '~icons/mdi/edit-outline';
  import MdiPlus from '~icons/mdi/plus';
  
  import type { TestItem, TestItemsContext } from '$lib/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.js';
  import EditTestItem from './EditTestItem.svelte';
  import AddTestItem from './AddTestItem.svelte';
	import { Skeleton } from "$lib/components/ui/skeleton/index.ts";

  let testItemsContext: TestItemsContext = getContext("testItemsContext");
  let allItems: TestItem[] = $state(testItemsContext.items);
  $effect(() => {
    allItems = testItemsContext.items;
  });

  let shortFormItems: TestItem[] = $derived(allItems.filter(item => !item.is_problem_solving));
  let probSolItems: TestItem[] = $derived(allItems.filter(item => item.is_problem_solving));
</script>


<div class="space-y-3 overflow-visible">
  <h1 class="text-left font-semibold">Test items</h1>
  {#if testItemsContext.isLoading}
    {#each { length: 2 } as _}
      <Skeleton class="h-24 w-full grayscale-100 rounded-none"/>
    {/each}
  {:else}
    {#each [{a: "Short Form Items", b: shortFormItems}, {a: "Problem-Solving Items", b: probSolItems}] as bigItem}
      <div class="card p-2 space-y-1">
        <span class="flex flex-row items-center w-full justify-between">
          <h4 class="font-medium">{ bigItem.a }</h4>
          <Dialog.Root>
            <Dialog.Trigger class="button-secondary">
              <MdiPlus class="size-5"/>
            </Dialog.Trigger>
            <AddTestItem test_id={data.test_id} />
          </Dialog.Root>
        </span>
        <div class="ml-2">
          {#each bigItem.b as smallItem}
            <span class="flex flex-row items-center justify-between gap-x-0.5">
              <p class="truncate text-ellipsis w-fill">
                ({smallItem.label}) {smallItem.question}
              </p>
              <Dialog.Root>
                <Dialog.Trigger>
                  <MdiEditOutline class="size-4"/>
                </Dialog.Trigger>
                <EditTestItem testItem={smallItem} test_id={data.test_id}/>
              </Dialog.Root>
            </span>
          {/each}
          {#if bigItem.b.length == 0}
            <p class="italic">Nothing to see here. If this is a mistake, check your network connection.</p>
          {/if}
        </div>
      </div>
    {/each}
  {/if}
</div>
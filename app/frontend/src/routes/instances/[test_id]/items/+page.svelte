<script lang="ts">
  const { data } = $props();

  import { getContext } from "svelte";
  
  import MdiEditOutline from '~icons/mdi/edit-outline';
  import MdiPlus from '~icons/mdi/plus';
  
  import type { TestItem, TestItemsContext } from '$lib/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.js';
  import { buttonVariants } from '$lib/components/ui/button/index.ts';
  import EditTestItem from './EditTestItem.svelte';
  import AddTestItem from './AddTestItem.svelte';
	import { Skeleton } from "$lib/components/ui/skeleton/index.ts";
	import Card from "$lib/components/Card.svelte";

  let isAddDialogOpen = $state(false);

  let testItemsContext: TestItemsContext = getContext("testItemsContext");
  let allItems: TestItem[] = $state(testItemsContext.items);
  $effect(() => {
    allItems = testItemsContext.items;
  });

  let shortFormItems: TestItem[] = $derived(
    allItems.filter(item => !item.is_problem_solving)
      .toSorted((a, b) => a.label.localeCompare(b.label))
    );
  let probSolItems: TestItem[] = $derived(
    allItems.filter(item => item.is_problem_solving)
      .toSorted((a, b) => a.label.localeCompare(b.label))
    );
</script>



<div class="space-y-3 overflow-visible">
  <h1 class="text-left font-semibold w-full flex justify-between items-center">
    Test items
    <Dialog.Root bind:open={isAddDialogOpen}>
      <Dialog.Trigger class={buttonVariants({ variant: 'secondary' })}>
        <MdiPlus class="size-5"/>
      </Dialog.Trigger>
      <AddTestItem {isAddDialogOpen}
                    test_id={data.test_id} />
    </Dialog.Root>
  </h1>

  {#if testItemsContext.isLoading}
    {#each { length: 2 } as _}
      <Skeleton class="h-24 w-full grayscale-100 rounded-none"/>
    {/each}
  
  {:else}
    {#each [{a: "Short Form Items", b: shortFormItems}, {a: "Problem-Solving Items", b: probSolItems}] as bigItem}
      <Card class="p-2 space-y-1">
        <h4 class="font-medium">{ bigItem.a }</h4>
        <div class="ml-2">
          {#each bigItem.b as smallItem}
            <Dialog.Root>
              <Dialog.Trigger class="flex flex-row items-center justify-between gap-x-0.5 w-full">
                <p class="text-left truncate text-ellipsis flex-1">
                  ({smallItem.label}) {smallItem.question}
                </p>
                <MdiEditOutline class="size-4"/>
              </Dialog.Trigger>
              <EditTestItem testItem={smallItem} test_id={data.test_id}/>
            </Dialog.Root>
          {/each}
          {#if bigItem.b.length == 0}
            <p class="italic">Nothing to see here. If this is a mistake, check your network connection.</p>
          {/if}
        </div>
      </Card>
    {/each}
  {/if}

</div>

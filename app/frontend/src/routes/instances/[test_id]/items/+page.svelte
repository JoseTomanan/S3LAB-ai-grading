<script lang="ts">
  import { getContext } from 'svelte';
  import type { TestItem, TestItemsContext } from '$lib/index.ts';
  import BottomSheet from '$lib/components/BottomSheet.svelte';
  import AddTestItem from './AddTestItem.svelte';
  import EditTestItem from './EditTestItem.svelte';
  import IconPlus from '~icons/mdi/plus';
  import IconPencil from '~icons/mdi/pencil-outline';

  const { data } = $props();

  let testItemsContext: TestItemsContext = getContext('testItemsContext');
  let allItems: TestItem[] = $state(testItemsContext.items);
  $effect(() => { allItems = testItemsContext.items; });

  let shortFormItems = $derived(
    allItems.filter((i) => !i.is_problem_solving).toSorted((a, b) => a.label.localeCompare(b.label))
  );
  let probSolItems = $derived(
    allItems.filter((i) => i.is_problem_solving).toSorted((a, b) => a.label.localeCompare(b.label))
  );

  let showAdd = $state(false);
  let editItem = $state<TestItem | null>(null);
</script>

<div class="p-4 flex flex-col gap-3">
  <!-- section header -->
  <div class="flex items-center justify-between">
    <span class="font-heading text-[20px] font-extrabold text-foreground tracking-[-0.04em]">
      Test Items
    </span>
    <button
      onclick={() => { showAdd = true; }}
      class="flex items-center gap-1.5 bg-primary text-white text-[13px] font-semibold px-3 py-1.5 rounded-md hover:opacity-90 transition-opacity"
    >
      <IconPlus class="size-4" /> Add
    </button>
  </div>

  <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
  {#each [{ label: 'Short Form', items: shortFormItems }, { label: 'Problem-Solving', items: probSolItems }] as group}
    <div class="bg-card rounded-xl shadow-sm px-3.5 py-3">
      <p class="text-[12px] font-bold uppercase tracking-[0.05em] text-foreground/55 mb-2.5">
        {group.label}
      </p>
      {#if group.items.length === 0}
        <p class="text-[13px] text-muted-foreground italic">No items yet.</p>
      {:else}
        <div class="flex flex-col">
          {#each group.items as item, i}
            {#if i > 0}
              <div class="border-t border-border my-2"></div>
            {/if}
            <div class="flex gap-2.5 items-start">
              <span class="font-mono text-[11px] text-foreground/55 mt-0.5 shrink-0 min-w-[16px]">
                {item.label}
              </span>
              <span class="flex-1 text-[13px] text-foreground leading-[1.45]">{item.question}</span>
              <span class="font-mono text-[11px] text-primary-700 font-bold mt-0.5 shrink-0">
                {item.expected_answer_rubric_questions.match(/\[(\d+)pts?\]/g)
                  ?.reduce((s, m) => s + parseInt(m.match(/\d+/)?.[0] ?? '0'), 0) ?? '?'}pt
              </span>
              <button
                onclick={() => { editItem = item; }}
                class="text-foreground/55 hover:text-foreground transition-colors mt-0.5 shrink-0"
              >
                <IconPencil class="size-[15px]" />
              </button>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/each}
  </div>
</div>

<BottomSheet bind:open={showAdd} title="New Test Item">
  {#snippet children()}
    <AddTestItem test_id={data.test_id} close={() => { showAdd = false; }} />
  {/snippet}
</BottomSheet>

<BottomSheet open={editItem !== null} title="Edit Item {editItem?.label ?? ''}">
  {#snippet children()}
    {#if editItem}
      <EditTestItem testItem={editItem} test_id={data.test_id} close={() => { editItem = null; }} />
    {/if}
  {/snippet}
</BottomSheet>

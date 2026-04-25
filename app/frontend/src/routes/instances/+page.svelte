<script lang="ts">
  import type { PageData } from './$types.ts';
  import type { TestInstance } from '$lib/index.ts';

  import TopBar from '$lib/components/TopBar.svelte';
  import BottomSheet from '$lib/components/BottomSheet.svelte';
  import { Button } from '$lib/components/CustomButton/index.ts';
  import Pagination from '$lib/components/Pagination.svelte';
  import AddTestInstance from './AddTestInstance.svelte';

  import IconPlus from '~icons/mdi/plus';
  import IconDone from '~icons/mdi/file-check-outline';
  import IconPending from '~icons/mdi/file-clock-outline';

  let { data }: { data: PageData } = $props();

  let instances: TestInstance[] = $derived(data.instances);
  let paginationValues: TestInstance[] = $state([]);
  let showAdd = $state(false);
</script>

<div class="flex flex-col min-h-full">
  {#snippet rightSlot()}
    <button
      onclick={() => showAdd = true}
      class="size-9 rounded-full bg-primary flex items-center justify-center text-white hover:opacity-90 transition-opacity"
    >
      <IconPlus class="size-[22px]" />
    </button>
  {/snippet}
  <TopBar title="Test Instances" right={rightSlot} />

  <div class="p-4">
    <div class="grid grid-cols-1 md:grid-cols-2 gap-2.5">
      {#each paginationValues as inst}
        <a
          href="/instances/{inst.test_id}/papers"
          class="bg-card rounded-xl shadow-sm px-4 py-3 flex items-center gap-3 no-underline hover:shadow-md transition-shadow"
        >
          <div class="size-[38px] rounded-[10px] flex items-center justify-center shrink-0
            {inst.is_done_rendering ? 'bg-secondary-300' : 'bg-accent'}">
            {#if inst.is_done_rendering}
              <IconDone class="size-5 text-secondary-700" />
            {:else}
              <IconPending class="size-5 text-primary-700" />
            {/if}
          </div>

          <div class="flex-1 min-w-0">
            <p class="text-[14px] font-medium text-foreground truncate opacity-100">{inst.name}</p>
            <p class="text-[11px] text-foreground/55 mt-0.5 font-mono opacity-100">{inst.test_id}</p>
          </div>

          <div class="flex flex-col items-end gap-1 shrink-0">
            <span class="text-[11px] text-foreground/55">
              {new Date(inst.date).toLocaleDateString()}
            </span>
            <span class="text-[12px] font-heading font-bold px-2 py-0.5 rounded-full
              {inst.is_done_rendering
                ? 'bg-secondary-300 text-secondary-700'
                : 'bg-muted text-muted-foreground'}">
              {inst.is_done_rendering ? 'Ready' : 'Pending'}
            </span>
          </div>
        </a>
      {/each}
    </div>

    {#if instances.length === 0}
      <p class="text-center text-muted-foreground text-sm py-8">No test instances yet.</p>
    {/if}

    <div class="mt-2.5 flex flex-col gap-2">
      <Button variant="add" onclick={() => { showAdd = true; }}>
        <IconPlus class="size-[18px]" />
        Add test instance
      </Button>
      <Pagination rows={instances} perPage={6} bind:trimmedRows={paginationValues} />
    </div>
  </div>
</div>

<BottomSheet bind:open={showAdd} title="New Test Instance">
  {#snippet children()}
    <AddTestInstance close={() => showAdd = false} />
  {/snippet}
</BottomSheet>

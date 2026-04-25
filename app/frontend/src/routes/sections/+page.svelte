<script lang="ts">
  import type { PageData } from './$types.ts';
  import type { Section } from '$lib/index.ts';
  import TopBar from '$lib/components/TopBar.svelte';
  import BottomSheet from '$lib/components/BottomSheet.svelte';
  import { Button } from '$lib/components/CustomButton/index.ts';
  import Pagination from '$lib/components/Pagination.svelte';
  import AddSection from './AddSection.svelte';
  import IconPlus from '~icons/mdi/plus';
  import IconClassroom from '~icons/mdi/google-classroom';
  import IconChevronRight from '~icons/mdi/chevron-right';

  let { data }: { data: PageData } = $props();

  let sections: Section[] = $derived(data.sections);
  let paginationValues: Section[] = $state([]);
  let showAdd = $state(false);
</script>

<div class="flex flex-col min-h-full">
  {#snippet rightSlot()}
    <button
      onclick={() => { showAdd = true; }}
      class="size-9 rounded-full bg-primary flex items-center justify-center text-white hover:opacity-90 transition-opacity"
    >
      <IconPlus class="size-[22px]" />
    </button>
  {/snippet}
  <TopBar title="Sections" right={rightSlot} />

  <div class="p-4">
    <div class="grid grid-cols-1 md:grid-cols-2 gap-2.5">
      {#each paginationValues as section}
        <a
          href="/sections/{section.section_id}"
          class="bg-card rounded-xl shadow-sm px-4 py-3 flex items-center gap-3 no-underline hover:shadow-md transition-shadow"
        >
          <div class="size-[38px] rounded-[10px] bg-accent flex items-center justify-center shrink-0">
            <IconClassroom class="size-5 text-foreground" />
          </div>
          <div class="flex-1 min-w-0">
            <p class="text-[14px] font-medium text-foreground truncate opacity-100">{section.section_name}</p>
          </div>
          <IconChevronRight class="size-5 text-foreground/55 shrink-0" />
        </a>
      {/each}
    </div>

    {#if sections.length === 0}
      <p class="text-center text-muted-foreground text-sm py-8">No sections yet.</p>
    {/if}

    <div class="mt-2.5 flex flex-col gap-2">
      <Button variant="add" onclick={() => { showAdd = true; }}>
        <IconPlus class="size-[18px]" />
        Add new section
      </Button>
      <Pagination rows={sections} perPage={6} bind:trimmedRows={paginationValues} />
    </div>
  </div>
</div>

<BottomSheet bind:open={showAdd} title="New Section">
  {#snippet children()}
    <AddSection close={() => { showAdd = false; }} />
  {/snippet}
</BottomSheet>

<script lang="ts">
  const { data } = $props();

  import MdiPeopleAdd from "~icons/mdi/people-add";
  import IconHome from '~icons/mdi/home';
  import IconPapers from '~icons/mdi/text-box-multiple-outline';

  import type { Section } from "$lib/index.ts";
  import { buttonVariants } from '$lib/components/ui/button/index.ts';
  import { cn } from '$lib/utils.ts';
  import Pagination from "$lib/components/Pagination.svelte";
  import * as Dialog from "$lib/components/ui/dialog/index.ts";
  import AddSection from "./AddSection.svelte";

  let sections: Section[] = $derived(data.sections);
  let paginationValues: Section[] = $state([]);
</script>


<nav class="w-full flex flex-row items-center justify-between mb-4
            px-4 pt-6">
  <a href="/" class={buttonVariants({ variant: 'floating' })}>
    <IconHome class="size-8"/>
  </a>
  <h1 class="italic">Sections</h1>
  <a href="/instances" class={buttonVariants({ variant: 'floating' })}>
    <IconPapers class="size-6.5 m-0.75"/>
  </a>
</nav>
<span class="-my-5"></span>

<div class="container">
  <div class="flex flex-col gap-3 relative">
    <Dialog.Root>
      <Dialog.Trigger class={cn(buttonVariants({ variant: 'outline' }), 'flex flex-row gap-x-2 items-center justify-center text-base *:font-semibold *:opacity-90')}>
        <MdiPeopleAdd class="size-5"/>
        <b>Add new section</b>
      </Dialog.Trigger>
      <AddSection />
    </Dialog.Root>
    
    {#if sections.length == 0}
      <div class="absolute top-0 right-0
                  flex flex-col items-center text-center mt-2 opacity-60">
        <p>No test instances yet. Tap above to add one!</p>
      </div>

    {:else}
      {#each paginationValues as section}
      <a href="/sections/{section.section_id}"
          class={cn(buttonVariants({ variant: 'outline' }), 'justify-between px-3 py-1.5 *:text-base')}>
        <span>{section.section_name}</span>
        <span class="font-normal font-mono opacity-60">
          {section.section_id}
        </span>
      </a>
      {/each}
    {/if}
  </div>
  
  <Pagination rows={sections}
              perPage={6}
              bind:trimmedRows={paginationValues} />
</div>
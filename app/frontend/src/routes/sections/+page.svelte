<script lang="ts">
  const { data } = $props();

  import MdiPeopleAdd from "~icons/mdi/people-add";
  import IconHome from '~icons/mdi/home-outline';
  import IconArrow from '~icons/mdi/arrow-up';
  import IconPapers from '~icons/mdi/paper';

  import type { Section } from "$lib/index.ts";
  import Pagination from "$lib/components/Pagination.svelte";
	import Card from "$lib/components/Card.svelte";
  import * as Dialog from "$lib/components/ui/dialog/index.ts";
  import AddSection from "./AddSection.svelte";

  let sections: Section[] = $derived(data.sections);
  let paginationValues: Section[] = $state([]);
</script>


<div class="container">
  <span class="flex flex-row items-center justify-between mb-4">
    <a href="/">
      <IconHome class="size-8"/>
    </a>
    <h1>Sections</h1>
    <a href="/instances">
      <IconPapers class="size-8 text-secondary saturate-150 brightness-60"/>
    </a>
  </span>
  <div class="flex flex-col gap-3 relative">
    <Dialog.Root>
      <Dialog.Trigger class="button-outline flex flex-row gap-x-2 justify-center 
                              *:opacity-90 *:font-semibold">
        <MdiPeopleAdd class="size-6"/>
        <b>Add new section</b>
      </Dialog.Trigger>
      <AddSection />
    </Dialog.Root>
    
    {#if sections.length == 0}
      <div class="absolute top-0 right-0
                  flex flex-col items-end text-right mt-2 opacity-60">
        <IconArrow class="size-10"/>
        <p>No test instances yet.<br>Tap here to add one!</p>
      </div>

    {:else}
      {#each paginationValues as section}
        <Card href="/sections/{section.section_id}"
              class="button-outline">
          <h3>{section.section_name}</h3>
          <h4>SectionID: {section.section_id}</h4>
        </Card>
      {/each}
    {/if}
  </div>
  
  <Pagination rows={sections}
              perPage={6}
              bind:trimmedRows={paginationValues} />
</div>
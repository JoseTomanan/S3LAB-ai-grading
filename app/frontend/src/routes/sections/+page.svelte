<script lang="ts">
	import { onMount } from "svelte";
	import MdiPapersOutline from "~icons/mdi/papers-outline";
	import MdiPeopleAdd from "~icons/mdi/people-add";
  import IconHome from '~icons/mdi/home-outline';
  import IconArrow from '~icons/mdi/arrow-up';

	import type { Section, Student } from "$lib/index.ts";
	import Pagination from "$lib/components/Pagination.svelte";
	import * as Dialog from "$lib/components/ui/dialog/index.ts";
	import { Skeleton } from "$lib/components/ui/skeleton/index.ts";
  import AddSection from "./AddSection.svelte";

	let isPageLoading: boolean = $state(true);
	let sections: Section[] = $state([]);
	let paginationValues: Section[] = $state([]);

	onMount(async () => {
		try {
			const response = await fetch(`/api/sections/`);

			switch (response.status) {
				case 200:
					const result = await response.json();
					sections = result;
					break;
				default:
					alert(`${response.status} ${response.statusText}`);
			}
		} catch(e) {
			alert("Failed to fetch sections:\nERROR: "+e);
		} finally {
			isPageLoading = false;
		}
	});
</script>


<div class="container">
	<span class="flex flex-row items-center justify-between mb-4">
    <span class="flex flex-row gap-x-2">
      <a href="/">
        <IconHome class="size-6"/>
      </a>
      <a href="/instances">
        <MdiPapersOutline class="size-6"/>
      </a>
    </span>
		<h1 class="italic">Sections</h1>
    <Dialog.Root>
      <Dialog.Trigger class="button-secondary">
        <MdiPeopleAdd class="size-6"/>
      </Dialog.Trigger>
      <AddSection />
    </Dialog.Root>
	</span>
	<div class="flex flex-col gap-3">
		{#if isPageLoading}
      {#each { length: 2 } as _}
        <Skeleton class="h-16 w-full grayscale-100"/>
      {/each}

		{:else if sections.length == 0}
			<div class="flex flex-col items-end text-right mt-2 opacity-60">
				<IconArrow />
				<p>No sections yet &mdash; tap here to add one!</p>
			</div>

		{:else}
			{#each paginationValues as section}
				<a href={`/sections/${section.section_id}`}
						class="card button-outline">
					<h3>{section.section_name}</h3>
					<h4>SectionID: {section.section_id}</h4>
				</a>
			{/each}
		{/if}
	</div>
	<Pagination rows={sections}
							perPage={6}
							bind:trimmedRows={paginationValues} />
</div>
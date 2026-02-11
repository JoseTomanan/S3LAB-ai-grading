<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	import { onMount } from "svelte";
	import MdiPapersOutline from "~icons/mdi/papers-outline";
	import MdiPersonAdd from "~icons/mdi/person-add";

	import type { Section, Student } from "$lib/index.ts";
	import Pagination from "$lib/components/Pagination.svelte";
	import * as Dialog from "$lib/components/ui/dialog/index.ts";

	let isPageLoading: boolean = false;
	let sections: Section[] = [{section_id: 1, section_name: "3-DavidAndal"}];
	let paginationValues: Section[];

	onMount(async () => {
	});
</script>


<div class="px-4 py-8 flex flex-col gap-4">
	<span class="flex flex-row items-center justify-between">
		<a href="/">
			<MdiPapersOutline class="size-8"/>
		</a>
		<h1>View sections</h1>
		<button class="button-primary" onclick={() => {}}>
			<MdiPersonAdd class="size-6"/>
		</button>
	</span>
	<div class="flex flex-col gap-3">
		{#if isPageLoading}
			<p>Loading...</p>
		{:else if sections.length == 0}
			<p>Nothing to see here. <br>Check your network connection, or add a new instance.</p>
		{:else}
			{#each paginationValues as section}
				<a href={`/sections/${section.section_id}`} class="card">
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
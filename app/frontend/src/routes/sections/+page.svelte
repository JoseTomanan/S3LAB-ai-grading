<script lang="ts">
	import { API_BASE_URL } from "$lib/constants.ts";

	import { onMount } from "svelte";
	import MdiPapersOutline from "~icons/mdi/papers-outline";
	import MdiPersonAdd from "~icons/mdi/person-add";

	import type { Section, Student } from "$lib/index.ts";
	import Pagination from "$lib/components/Pagination.svelte";
	import * as Dialog from "$lib/components/ui/dialog/index.ts";

	let isPageLoading: boolean = $state(true);
	let sections: Section[] = $state([]);
	let paginationValues: Section[] = $state([]);

	onMount(async () => {
		try {
			const response = await fetch(
						`${API_BASE_URL}/api/sections`,
						{
							method: "GET",
							headers: {'Content-Type': 'application/json',},
						}
						);

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


<div class="container gap-8">
	<span class="flex flex-row items-center justify-between">
		<a href="/">
			<MdiPapersOutline class="size-6"/>
		</a>
		<h1 class="italic">Sections</h1>
		<button class="button-secondary" onclick={() => {}}>
			<MdiPersonAdd class="size-6"/>
		</button>
	</span>
	<div class="flex flex-col gap-4">
		{#if isPageLoading}
			<p>Loading...</p>
		{:else if sections.length == 0}
			<p>Nothing to see here. <br>Check your network connection, or add a new instance.</p>
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
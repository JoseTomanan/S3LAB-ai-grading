<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	import { onMount } from 'svelte';
	import MdiPaperAddOutline from '~icons/mdi/paper-add-outline';
	
	import type { TestInstance } from '$lib/types/types.ts';
	
	import Pagination from '$lib/components/Pagination.svelte';
	import TestInstanceCard from './TestInstanceCard.svelte';
	import * as Dialog from "$lib/components/ui/dialog/index.js";
	import AddTestInstance from './AddTestInstance.svelte';
	
	let isPageLoading: boolean = true;
	let instances: TestInstance[] = [];
	let paginationValues: TestInstance[];

	onMount(async () => {
		try {
			const response = await fetch(
				`${apiBaseUrl}/api/test_instances`,
				{
					method: "GET",
					headers: {'Content-Type': 'application/json',}
				}
			);

			const data = await response.json();
			instances = data.instances;
		} catch (e) {
			alert("Failed to fetch instances:\n"+e);
		} finally {
			isPageLoading = false;
		}
	});
</script>


<div class="px-4 py-8 flex flex-col gap-4">
	<span class="flex flex-row items-center justify-between">
		<h1>View test instances</h1>
		<Dialog.Root>
			<Dialog.Trigger class="button-primary">
				<MdiPaperAddOutline class="size-6"/>
			</Dialog.Trigger>
			<AddTestInstance />
		</Dialog.Root>
	</span>
	<div class="flex flex-col gap-3">
		{#if isPageLoading}
			<p>Loading...</p>
		{:else if instances.length == 0}
			<p>Nothing to see here. <br>Check your network connection, or add a new instance.</p>
		{:else}
			{#each paginationValues as instance}
				<TestInstanceCard {...instance}/>
			{/each}
		{/if}
	</div>
	<Pagination rows={instances}
							perPage={6}
							bind:trimmedRows={paginationValues} />
</div>

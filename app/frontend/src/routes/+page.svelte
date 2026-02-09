<script lang="ts">
	import { API_BASE_URL } from '$lib/constants.ts';
	import { onMount } from 'svelte';

	import MdiCheckboxBlankOutline from '~icons/mdi/checkbox-blank-outline';
	import MdiCheckboxMarkedOutline from '~icons/mdi/checkbox-marked-outline';
	import MdiPaperAddOutline from '~icons/mdi/paper-add-outline';
	import MdiGoogleClassroom from '~icons/mdi/google-classroom';
	
	import type { TestInstance } from '$lib/index.ts';
	
	import Pagination from '$lib/components/Pagination.svelte';
	import * as Dialog from "$lib/components/ui/dialog/index.js";
	import AddTestInstance from './AddTestInstance.svelte';
	
	let isPageLoading: boolean = true;
	let instances: TestInstance[] = [];
	let paginationValues: TestInstance[];

	onMount(async () => {
		try {
			const response = await fetch(
				`${API_BASE_URL}/api/test_instances`,
				{
					method: "GET",
					headers: {'Content-Type': 'application/json',}
				}
			);

			switch (response.status) {
				case 200:
					const result = await response.json();
					console.log(result);
					if (result.instances)
						instances = result.instances;
					break;
				default:
					alert(`${response.status} ${response.statusText}`);
			}
		} catch (e) {
			alert("Failed to fetch instances:\n"+e);
		} finally {
			isPageLoading = false;
		}
	});
</script>


<div class="px-4 py-8 flex flex-col gap-4">
	<span class="flex flex-row items-center justify-between">
		<a href="/sections">
			<MdiGoogleClassroom class="size-8"/>
		</a>
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
				<a href={`/${instance.test_id}/items`} class="card">
					<span class="flex flex-row items-center gap-1">
						{#if instance.is_done_rendering}
							<MdiCheckboxMarkedOutline/>
						{:else}
							<MdiCheckboxBlankOutline/>
						{/if}
						<h3>{ instance.name }</h3>
					</span>
					<h4>{ new Date(instance.date).toLocaleDateString() }</h4>
				</a>
			{/each}
		{/if}
	</div>
	<Pagination rows={instances}
							perPage={6}
							bind:trimmedRows={paginationValues} />
</div>

<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	import { onMount } from 'svelte';
	import MdiPlus from '~icons/mdi/plus';
	
	import type { TestInstance } from '$lib/types/types.ts';
	
	import TestInstanceCard from './TestInstanceCard.svelte';
	import Pagination from './Pagination.svelte';
	import * as Dialog from "$lib/components/ui/dialog/index.js";
	import { Label } from '$lib/components/ui/label/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Button } from '$lib/components/ui/button/index.ts';
	
	let isPageLoading: boolean = true;
	let instances: TestInstance[] = [];
	let paginationValues: TestInstance[];

	let newInstanceName: string = "";
	let newInstanceSection: string = "";

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

	async function addNewTestInstance(name: string, section: string) {
		// TODO: finish (remove this only when verified with working API)
		try {
			const response = await fetch(
				`${apiBaseUrl}/api/test_instances`,
				{
					method: "POST",
					headers: {'Content-Type': 'application/json',},
					body: JSON.stringify({
						"name": name,
						"section": section,
						"date": new Date().toISOString(),
					}),
				}
			);

			if (response.status == 200) {
				const data = response.json();
				alert("Addition successful.");
				window.location.reload();
			} else {
				alert("Addition fail: " + response.status + response.statusText);
			}
		} catch (e) {
			alert("Failed to add new test instance. Check your network connection and try again.");
		}
	}

</script>

<div class="px-4 py-8 flex flex-col gap-4">
	<span class="flex flex-row items-center justify-between">
		<h1>View Test Instances</h1>
		<Dialog.Root>
			<Dialog.Trigger>
				<MdiPlus class="size-8"/>
			</Dialog.Trigger>
			<Dialog.Content>
				<Dialog.Header>
					<Dialog.Title>Add new test instance</Dialog.Title>
				</Dialog.Header>
				<Label for="name">Test name</Label>
				<Input id="name" type="text" placeholder="Quiz 1"
								bind:value={newInstanceName}/>
				<Label for="section">Section</Label>
				<Input id="section" type="text" placeholder="1-Acacia"
								bind:value={newInstanceSection}/>
				<Dialog.Footer>
					<Button variant="outline"
									onclick={() => addNewTestInstance(newInstanceName, newInstanceSection)}>
						Save changes
				</Button>
					<Dialog.Description>Note that the name and section cannot be changed after creation.</Dialog.Description>
				</Dialog.Footer>
			</Dialog.Content>
		</Dialog.Root>
	</span>
	<div class="flex flex-col gap-3">
		{#if isPageLoading}
			<p>Loading...</p>
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

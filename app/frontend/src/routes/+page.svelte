<script lang="ts">
	import { onMount } from 'svelte';
	import MdiPlus from '~icons/mdi/plus';
	
	import type { TestInstance } from '$lib/types/types.ts';
	
	import TestInstanceCard from '$lib/components/cards/TestInstanceCard.svelte';
	import Pagination from '$lib/components/Pagination.svelte';
	import * as Dialog from "$lib/components/ui/dialog/index.js";
	import { Label } from '$lib/components/ui/label/index.js';
	import { Input } from '$lib/components/ui/input/index.js';

	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
	
	let isPageLoading: boolean = true;
	let instances: TestInstance[] = [];
	let paginationValues: TestInstance[];

	// untested, TODO : verify if correct logic
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
			instances = data.map((item: TestInstance) => item);
		} catch (e) {
			console.log("Failed to fetch instances:\n"+e);
		} finally {
			isPageLoading = false;
		}
	});

	let newInstanceName: string = "";
	let newInstanceSection: string = "";

	async function addNewTestInstance(name: string, section: string) {
		// TODO : finish (remove this only when verified with working API)
		try {
			const response = await fetch(
				`${apiBaseUrl}/api/test_instances`,
				{
					method: "POST",
					headers: {'Content-Type': 'application/json',},
					body: JSON.stringify({
						"name": name,
						"section": section
					}),
				}
			);

			if (response.status == 200) {
				const data = response.json();
				alert("Addition of instance with ID successful:\n"+data);
			} else {
				alert("Addition fail: " + response.statusText);
			}
		} catch (e) {
			alert("Failed to add new test instance:\n"+e);
		}
	}

	// TODO : remove once API is working
	instances = [
		{test_name: "Seatwork-1", section: "3-Rizal", date: "2025-11-11T20:17:46.384Z", is_done_rendering: true},
		{test_name: "Seatwork-2", section: "3-Aguinaldo", date: "2025-12-12T20:17:46.384Z", is_done_rendering: false},
		{test_name: "Quiz-1", section: "3-Aguinaldo", date: "2026-01-12T20:17:46.384Z", is_done_rendering: false},
		{test_name: "Quiz-1", section: "3-Rizal", date: "2026-01-13T20:17:46.384Z", is_done_rendering: true},
		{test_name: "Quiz-2", section: "3-Aguinaldo", date: "2026-01-19T20:17:46.384Z", is_done_rendering: true},
	];

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
				<Label for="name">Name</Label>
				<Input id="name" type="text" placeholder="Quiz #1"
							bind:value={newInstanceName}/>
				<Label for="section">Section</Label>
				<Input id="section" type="text" placeholder="1-Acacia"
							bind:value={newInstanceSection}/>
				<Dialog.Footer>
					<button class="button-primary"
								on:click={() => addNewTestInstance(newInstanceName, newInstanceSection)}>
						Save changes
					</button>
					<Dialog.Description>Note that test name and section cannot be changed after creation.</Dialog.Description>
				</Dialog.Footer>
			</Dialog.Content>
		</Dialog.Root>
	</span>
	<div class="flex flex-col gap-3">
		{#each paginationValues as instance}
			<TestInstanceCard {...instance} />
		{/each}
	</div>
	<Pagination
			rows={instances}
			perPage={6}
			bind:trimmedRows={paginationValues} />
</div>

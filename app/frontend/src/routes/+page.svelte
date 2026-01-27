<script lang="ts">
	import { onMount } from 'svelte';
	import MdiPlus from '~icons/mdi/plus';

	import TestInstanceCard from '$lib/components/cards/TestInstanceCard.svelte';
	import Pagination from '$lib/components/Pagination.svelte';
	import type { TestInstance } from '$lib/types/types.ts';
	import { error } from 'console';

	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
	
	let isPageLoading: boolean = true;
	let instances: TestInstance[] = [];
	let paginationValues: TestInstance[];

	// unstable; TODO : verify if correct logic
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
			instances = data.map((item: TestInstance) => {
				return item
			});
		} catch (e) {
			alert("Failed to fetch instances.");
		} finally {
			isPageLoading = false;
		}
	});

	// temporary!
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
		<button>
			<!-- TODO : open imported dialog component as popup on click -->
			<MdiPlus class="size-8"/>
		</button>
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

<script lang="ts">
	import TestInstanceCard from '$lib/components/TestInstanceCard.svelte';
	import Pagination from '$lib/components/Pagination.svelte';

	import MdiPlus from '~icons/mdi/plus';

	import type { TestInstance } from '$lib/types/types.ts';
	
	let instances: TestInstance[] = [
		{test_name: "Seatwork-1", section: "3-Rizal", date: "2025-11-11T20:17:46.384Z", is_done_rendering: true},
		{test_name: "Seatwork-2", section: "3-Aguinaldo", date: "2025-12-12T20:17:46.384Z", is_done_rendering: false},
		{test_name: "Quiz-1", section: "3-Aguinaldo", date: "2026-01-12T20:17:46.384Z", is_done_rendering: false},
		{test_name: "Quiz-1", section: "3-Rizal", date: "2026-01-13T20:17:46.384Z", is_done_rendering: true},
		{test_name: "Quiz-2", section: "3-Aguinaldo", date: "2026-01-19T20:17:46.384Z", is_done_rendering: true},
	];

	let paginationValues: TestInstance[];
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
	<hr>
	<Pagination
			rows={instances}
			perPage={5}
			bind:trimmedRows={paginationValues} />
</div>

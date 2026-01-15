<script lang="ts">
	import type { TestInstance } from '$lib/types.ts';
	
	let instances: TestInstance[] = [
		{test_name: "Quiz 1", section: "3-Aguinaldo", date: "2026-01-12T20:17:46.384Z", is_done_rendering: true},
		{test_name: "Quiz 2", section: "3-Aguinaldo", date: "2025-12-12T20:17:46.384Z", is_done_rendering: false},
	];
</script>

<h1>View test instances</h1>

<div class="flex flex-col gap-6">
	{#each instances as instance}
		<pre>{JSON.stringify(instance, null, 2)}</pre>
	{/each}
</div>

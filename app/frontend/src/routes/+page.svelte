<script lang="ts">
	import type { TestInstance } from '$lib/types';
	
	let instances: TestInstance[] = [
		{test_name: "Quiz 1", section: "3-Aguinaldo", date: "kasbu erp", is_done_rendering: true},
		{test_name: "Quiz 2", section: "3-Aguinaldo", date: "next week erp", is_done_rendering: false},
	];
</script>

<h1>View test instances</h1>

<div class="flex flex-col gap-6">
	{#each instances as instance}
		<pre>{JSON.stringify(instance, null, 2)}</pre>
	{/each}
</div>

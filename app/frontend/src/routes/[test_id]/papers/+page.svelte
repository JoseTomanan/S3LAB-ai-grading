<script lang="ts">
	const { data } = $props();
		
	import { onMount } from "svelte";
	import MdiPaperAddOutline from '~icons/mdi/paper-add-outline';

	import type { Student, TestItemHeader } from "$lib/types/types.ts";

	let isPageLoading: boolean = $state(true);
	let students: Student[] = $state([]);
	let itemHeaders: TestItemHeader[] = $state([]);

	onMount(async () => {
		// TODO: finish; revert console.log() to alert() and isPageLoading when done.
		try {
			const response = await fetch(``, {});
		} catch(e) {
			console.log("Failed to fetch, check your network connection and try again.\n"+e);
		} finally {
		}
	});
</script>


<div class="space-y-4">
	{#if isPageLoading}
		<p>Loading...</p>
	{:else}
		{#each students as student}
			<a href={`/${data.test_id}/papers/${student.student_no}`}>
				<div class="card flex flex-row gap-2 items-center">
					<MdiPaperAddOutline class="size-6"/>
					<h3>{student.name}</h3>
				</div>
			</a>
		{/each}
	{/if}
</div>
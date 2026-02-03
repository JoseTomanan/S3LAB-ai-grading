<script lang="ts">
	const { data } = $props();
		
	import { onMount } from "svelte";
	import MdiPaperEditOutline from '~icons/mdi/paper-edit-outline';

	import type { Student, TestItemHeader } from "$lib/types/types.ts";

	let isPageLoading: boolean = $state(true);
	let students: Student[] = $state([]);

	onMount(async () => {
		// TODO: finish; revert console.log() to alert() and isPageLoading when done.
		try {
			const response = await fetch(``, {});
		} catch(e) {
			console.log("Failed to fetch, check your network connection and try again.\n"+e);
		} finally {
		}
	});

	// FIXME: placeholders; remove when done
	setTimeout(() => {
		students = [{student_no: "202100000", name: "Abalos, Benhur P.", section: "1-Narra"},
					{student_no: "202100022", name: "Baclaan, Kean Christopher P.", section: "1-Narra"}];
		isPageLoading = false;
	}, 500);
</script>


<div class="space-y-4">
	{#if isPageLoading}
		<p>Loading...</p>
	{:else}
		{#each students as student}
			<a href={`/${data.test_id}/papers/${student.student_no}`}
						class="card flex flex-row items-center justify-between">
				<h3>{student.name}</h3>
				<MdiPaperEditOutline class="size-6"/>
			</a>
		{/each}
	{/if}
</div>
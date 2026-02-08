<script lang="ts">
	const { data } = $props();
	
	import { API_BASE_URL } from '$lib/constants.ts';
	import { onMount } from "svelte";
	
	import MdiPaperEditOutline from '~icons/mdi/paper-edit-outline';
	import MdiPaperCheckOutline from '~icons/mdi/paper-check-outline';

	let isPageLoading: boolean = $state(true);
	let perStudentStatuses: {student_no: string, name: string, is_done_rendering: boolean}[] = $state([]);

	onMount(async () => {
		try {
			const response = await fetch(
						`${API_BASE_URL}/api/test_instances/${data.test_id}/statuses`,
						{
							method: "GET",
							headers: {'Content-Type': 'application/json',},
						}
					);

			const result = await response.json();

			if (!response.ok)
				alert(`${response.status} ${response.statusText}`);
			else
				perStudentStatuses = result.statuses;
		} catch(e) {
			alert("Failed to fetch, check your network connection and try again.\nERROR: "+e);
		} finally {
			isPageLoading = false;
		}
	});
</script>


<div class="space-y-4">
	{#if isPageLoading}
		<p>Loading...</p>
	{:else if perStudentStatuses.length == 0}
		<p>Nothing to see here. <br>If this is a mistake, check your network connection.</p>
	{:else}
		{#each perStudentStatuses as status}
			<a href={`/${data.test_id}/papers/${status.student_no}`}
						class="card flex flex-row items-center justify-between">
				<h3>{status.name}</h3>
				{#if status.is_done_rendering}
					<MdiPaperEditOutline class="size-6"/>
				{:else}
					<MdiPaperCheckOutline class="size-6"/>
				{/if}
			</a>
		{/each}
	{/if}
</div>
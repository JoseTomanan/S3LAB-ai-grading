<script lang="ts">
	const { data } = $props();
	
	import { API_BASE_URL } from '$lib/constants.ts';
	import { onMount } from "svelte";
	
	import MdiPaperAlertOutline from '~icons/mdi/paper-alert-outline';
	import MdiPaperCheckOutline from '~icons/mdi/paper-check-outline';

	import * as Dialog from '$lib/components/ui/dialog/index.ts';

	import GetAIEvaluation from './GetAIEvaluation.svelte';

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

			switch (response.status) {
				case 200:
					const result = await response.json();
					perStudentStatuses = result.statuses;
					break;
				default:
					alert(`${response.status} ${response.statusText}`);
			}
		} catch(e) {
			alert("Failed to fetch test paper statuses:\nERROR: "+e);
		} finally {
			isPageLoading = false;
		}
	});
</script>


<div class="space-y-3">
	{#if isPageLoading}
		<p>Loading...</p>
	{:else if perStudentStatuses.length == 0}
		<p>Nothing to see here. <br>If this is a mistake, check your network connection.</p>
	{:else}
		{#each perStudentStatuses as testPaper}
			<span class="flex flex-row items-center justify-between gap-2">
				<a href={`/instances/${data.test_id}/papers/${testPaper.student_no}`}
						class="card button-outline w-5/6">
					<h3>{testPaper.name}</h3>
				</a>
				<Dialog.Root>
					<Dialog.Trigger class="w-1/6 button-outline flex items-center justify-center">
						{#if testPaper.is_done_rendering}
							<MdiPaperCheckOutline class="size-6 my-0.5"/>
						{:else}
							<MdiPaperAlertOutline class="size-6 my-0.5"/>
						{/if}
					</Dialog.Trigger>
					<GetAIEvaluation test_id={data.test_id}
														student_no={testPaper.student_no}/>
				</Dialog.Root>
			</span>
		{/each}
	{/if}
</div>

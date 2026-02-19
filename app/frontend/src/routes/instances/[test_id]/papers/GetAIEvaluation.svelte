<script lang="ts">
	const { test_id, student_no } = $props();

	import { API_BASE_URL } from '$lib/constants.ts';
	import { onMount } from "svelte";

	import type { EvaluationsResponse, StudentStoresResponse } from '$lib/index.ts';
	import * as Dialog from "$lib/components/ui/dialog/index.ts";

	let isPageLoading = $state(false);
	let questionItemEvals: EvaluationsResponse[] = $state([]);

	const GET_E_A_R_Q = (i: EvaluationsResponse) => i.expected_answer_rubric_questions.split(';');
	const GET_EVALS = (i: EvaluationsResponse) => i.ai_evaluation.split(';');

	onMount(async () => {
		isPageLoading = true;
		try {
			const response = await fetch(
						`${API_BASE_URL}/api/test_instances/${test_id}/results/${student_no}`,
						{
							method: "GET",
							headers: {'Content-Type': 'application/json',},
						}
						);

			switch (response.status) {
				case 200:
					const result = await response.json();
					questionItemEvals = result.evaluations;
					break;
				default:
					alert(`${response.status} ${response.statusText}`);
			}
		} catch (e) {
			alert("Failed to fetch, check your network connection and try again.\nERROR: "+e);
		} finally {
			isPageLoading = false;
		}
	});
</script>


<Dialog.Content>
	<Dialog.Header>
		<Dialog.Title>AI evaluation results</Dialog.Title>
		<Dialog.Description>{test_id} &middot; {student_no}</Dialog.Description>
	</Dialog.Header>
	<div class="max-h-[80vh] overflow-auto space-y-2">
		{#each questionItemEvals as evalItem}
			<div class="card space-y-1">
				<h4 class="truncate text-ellipsis w-fill">
					{evalItem.label}: {evalItem.question}
				</h4>
				{#each GET_E_A_R_Q(evalItem) as e_a_r_q, index}
					{#if e_a_r_q.length != 0}
						<span class="flex flex-row justify-between">
							<h6 class="italic">{e_a_r_q}</h6>
							<h6 class="font-bold">{GET_EVALS(evalItem)[index]}</h6>
						</span>
					{/if}
				{/each}
			</div>
		{/each}
	</div>
</Dialog.Content>
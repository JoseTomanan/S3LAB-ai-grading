<script lang="ts">
	const { test_id, student_no } = $props();

	import { API_BASE_URL } from '$lib/constants.ts';
	import { onMount } from "svelte";

	import MdiHeadReload from "~icons/mdi/head-reload";

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
			alert("Failed to fetch AI evaluations for given student:\n- ERROR: "+e);
		} finally {
			isPageLoading = false;
		}
	});

	async function reevaluateAnswer(answer_id: number) {
		isPageLoading = true;
		try {
			const response = await fetch(
						`${API_BASE_URL}/api/answers/${answer_id}/reevaluate`,
						{
							method: "PATCH",
							headers: {'Content-Type': 'application/json',},
						}
						);

			switch (response.status) {
				case 200:
					const result = await response.json();
					questionItemEvals = questionItemEvals.map(ans => 
									ans.answer_id === result.answer_id 
									? { ...ans, ai_evaluation: result.ai_evaluation }
									: ans
								);
					break;
				default:
					alert(`${response.status} ${response.statusText}`);
			}
		} catch (e) {
			alert("Failed to reevaluate answer for given answer_id:\n- ERROR: "+e);
		} finally {
			isPageLoading = false;
		}
	}
</script>


<Dialog.Content>
	<Dialog.Header>
		<Dialog.Title>AI evaluation results</Dialog.Title>
		<Dialog.Description>{test_id} &middot; {student_no}</Dialog.Description>
	</Dialog.Header>
	<div class="max-h-[80vh] overflow-auto space-y-2">
		{#if questionItemEvals.length == 0}
			<p>
				Nothing to see here. Please upload images of student answers first.
			</p>
		{:else if isPageLoading}
			<p>Loading...</p>
		{:else}
			{#each questionItemEvals as evalItem}
				<div class="card space-y-1">
					<span class="flex flex-row justify-between items-center gap-x-2">
						<h4 class="truncate text-ellipsis w-fill">
							{evalItem.label}: {evalItem.question}
						</h4>
						<button class="button-secondary px-0 py-0"
										onclick={() => reevaluateAnswer(evalItem.answer_id)}>
							<MdiHeadReload />
						</button>
					</span>
					{#each GET_E_A_R_Q(evalItem) as e_a_r_q, index}
						{#if e_a_r_q.length != 0}
							<span class="flex flex-row justify-between">
								<h6 class="italic">{e_a_r_q}</h6>
								{#if GET_EVALS(evalItem)[index] != ""}
									<h6 class="font-bold">
										{GET_EVALS(evalItem)[index]}
									</h6>
								{:else}
									<h6>&mdash;</h6>
								{/if}
							</span>
						{/if}
					{/each}
				</div>
			{/each}
		{/if}
	</div>
</Dialog.Content>
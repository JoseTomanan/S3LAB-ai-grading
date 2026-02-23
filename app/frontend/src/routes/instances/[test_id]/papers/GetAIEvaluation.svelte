<script lang="ts">
	const { test_id, student_no } = $props();

	import { API_BASE_URL } from '$lib/constants.ts';
	import { getContext, onMount } from "svelte";

	import MdiHeadReload from "~icons/mdi/head-reload";
	import MdiAlertBoxOutline from "~icons/mdi/alert-box-outline";

	import type { GetSpecificEvaluationResponse, TestItem, TestItemsContext } from '$lib/index.ts';
	import * as Dialog from "$lib/components/ui/dialog/index.ts";

	let testItemsContext: TestItemsContext = getContext("testItemsContext");

	let isPageLoading = $state(false);
	let testItems: TestItem[] = $state(testItemsContext.items);

	let questionItemEvals: GetSpecificEvaluationResponse[] = $state([]);

	const GET_E_A_R_Q = (i: GetSpecificEvaluationResponse) => i.expected_answer_rubric_questions.split(';');
	const GET_EVALS = (i: GetSpecificEvaluationResponse) => i.ai_evaluation.split(';');
	const GET_SCORES = (i: GetSpecificEvaluationResponse) => i.scores.split(';');

	const REPOPULATE_UNANSWERED_ITEMS = () => {
					const evalItemIds = new Set(questionItemEvals.map(e => e.item_id));
					const unansweredItems = testItems
									.filter(item => !evalItemIds.has(item.item_id))
									.map(item => ({
										answer_id: -1,
										item_id: item.item_id,
										label: item.label,
										question: item.question,
										expected_answer_rubric_questions: item.expected_answer_rubric_questions,
										ai_evaluation: "",
										scores: "",
									}));
					questionItemEvals.push(...unansweredItems);
				};

	onMount(async () => {
		isPageLoading = true;
		REPOPULATE_UNANSWERED_ITEMS();
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
					REPOPULATE_UNANSWERED_ITEMS();
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
					REPOPULATE_UNANSWERED_ITEMS();
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
		{#if isPageLoading}
			<p>Loading...</p>
		{:else}
			{#each questionItemEvals as evalItem}
				<div class="card space-y-1">
					{#if evalItem.answer_id == -1}
						<h4 class="truncate text-ellipsis w-fill">
							{evalItem.label}: {evalItem.question}
						</h4>
						<h6 class="flex flex-row items-center gap-1">
							<MdiAlertBoxOutline />
							This item has no corresponding answer yet.
						</h6>
					{:else}
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
								<!--
								{@const answerEval = GET_EVALS(evalItem)[index]}
								{@const isHasAnswer = answerEval && answerEval != ""}
								-->
								{@const answerScore = GET_SCORES(evalItem)[index]}
								{@const isHasScore = answerScore && answerScore != ""}
								<span class="flex flex-row justify-between">
									<h6 class="italic">{e_a_r_q}</h6>
									<!--
									<h6 class={isHasAnswer ? "font-bold" : ""}>
										{isHasAnswer ? answerEval : "—"}
									</h6>
									-->
									<h6 class={isHasScore ? "font-bold" : ""}>
										{isHasScore ? answerScore : "—"}
									</h6>
								</span>
							{/if}
						{/each}
					{/if}
				</div>
			{/each}
		{/if}
	</div>
</Dialog.Content>
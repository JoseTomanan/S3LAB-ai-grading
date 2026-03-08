<script lang="ts">
  const { test_id, student_no } = $props();

  import { API_BASE_URL } from '$lib/constants.ts';
  import { getContext, onMount } from "svelte";

  import MdiHeadReload from "~icons/mdi/head-reload";
  import MdiAlertBoxOutline from "~icons/mdi/alert-box-outline";

  import type { GetSpecificEvaluationResponse, TestItem, TestItemsContext } from '$lib/index.ts';
  import * as Dialog from "$lib/components/ui/dialog/index.ts";
	import Skeleton from '$lib/components/ui/skeleton/skeleton.svelte';
	import { Spinner } from '$lib/components/ui/spinner/index.ts';
	import { Item } from '$lib/components/ui/dropdown-menu/index.ts';

  let testItemsContext: TestItemsContext = getContext("testItemsContext");

  let isPageLoading = $state(false);
  let testItems: TestItem[] = $state(testItemsContext.items);

  let questionItemEvals: GetSpecificEvaluationResponse[] = $state([]);
  let isRequestOngoings: Map<number, boolean> = $state(new Map());

  const GET_E_A_R_Q = (i: GetSpecificEvaluationResponse) => i.expected_answer_rubric_questions.split(';');
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
          unansweredItems.forEach(i => isRequestOngoings.set(i.answer_id, false));
        };

  onMount(async () => {
    isPageLoading = true;
    REPOPULATE_UNANSWERED_ITEMS();
    try {
      const response = await fetch(
            `${API_BASE_URL}/api/student_answers/${test_id}/results/${student_no}`,
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
    isRequestOngoings = new Map(isRequestOngoings.set(answer_id, true));
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
                  ans.answer_id == result.answer_id 
                  ? { ...ans, ai_evaluation: result.ai_evaluation, scores: result.scores }
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
      isRequestOngoings = new Map(isRequestOngoings.set(answer_id, false));
    }
  }

  const totalScore: string = $derived((() => {
    let sumNumerator = 0;
    let sumDenominator = 0;

    for (const evalItem of questionItemEvals) {
      const scores = GET_SCORES(evalItem);
      for (const score of scores) {
        if (typeof score === "string" && score.includes("/")) {
          const [num, denom] = score.split("/");
          const n = parseFloat(num);
          const d = parseFloat(denom);
          if (!isNaN(n))
            sumNumerator += n;
          if (!isNaN(d))
            sumDenominator += d;
        }
      }
    }

    if (sumNumerator === 0 && sumDenominator === 0)
      return "·/·";
    return `${sumNumerator}/${sumDenominator}`;
  })());
</script>


<Dialog.Root>
  <Dialog.Trigger class="w-1/6 button-outline flex items-center justify-center">
    <h4 class="font-semibold opacity-80 tracking-wide">
      {totalScore ?? "n/a"}
    </h4>
  </Dialog.Trigger>

<Dialog.Content>
  <Dialog.Header>
    <Dialog.Title>AI evaluation results</Dialog.Title>
    <Dialog.Description>{test_id} &middot; {student_no}</Dialog.Description>
  </Dialog.Header>
  <div class="max-h-[80vh] overflow-auto space-y-2">
    {#if isPageLoading}
      {#each { length: 3 } as _}
        <Skeleton class="h-12 w-full grayscale-100 rounded-none"/>
      {/each}
    {:else}
      {#each questionItemEvals.slice().sort((a, b) => a.label.localeCompare(b.label)) as evalItem}
        <div class="card space-y-1">
          {#if evalItem.answer_id == -1}
            <h4 class="truncate text-ellipsis w-fill">
              {evalItem.label}: {evalItem.question}
            </h4>
            <h6 class="flex flex-row items-center gap-1">
              <MdiAlertBoxOutline />
              Student has no uploaded answer yet.
            </h6>

          {:else}
            {@const isRequestLoading = isRequestOngoings.get(evalItem.answer_id)}
            {@const e_a_r_qs = GET_E_A_R_Q(evalItem)}
            <span class="flex flex-row justify-between items-center gap-x-2">
              <h4 class="truncate text-ellipsis w-fill">
                {evalItem.label}: {evalItem.question}
              </h4>
              <button class={`${isRequestLoading ? "opacity-50" : "opacity-100"} button-secondary px-0 py-0`}
                      onclick={() => reevaluateAnswer(evalItem.answer_id)}
                      disabled={isRequestLoading}>
                <MdiHeadReload />
              </button>
            </span>
            {#each e_a_r_qs as e_a_r_q, index}
              {#if e_a_r_q.length != 0}
                {@const answerScore = GET_SCORES(evalItem)[index]}
                {@const isHasScore = answerScore && answerScore != ""}
                <span class="flex flex-row justify-between items-center">
                  <h6 class="italic">{e_a_r_q}</h6>
                  {#if isRequestLoading}
                    <Spinner class="size-4" />
                  {:else}
                    <h6 class={isHasScore ? "font-bold" : ""}>
                      {isHasScore ? answerScore : "—"}
                    </h6>
                  {/if}
                </span>
              {/if}
            {/each}
          {/if}
        </div>
      {/each}
    {/if}
  </div>
</Dialog.Content>

</Dialog.Root>
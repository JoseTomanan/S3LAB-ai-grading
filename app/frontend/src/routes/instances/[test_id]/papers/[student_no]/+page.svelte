<script lang="ts">
  const { data } = $props();

  
  import MdiPaperOff from '~icons/mdi/paper-off';
  import MdiImagePlus from '~icons/mdi/image-plus';
  import MdiCrop from '~icons/mdi/crop';
  import IconReevaluate from "~icons/mdi/head-reload";

  import type { GetSpecificEvaluationResponse, StudentAnswer } from '$lib/index.ts';
  import { Label } from '$lib/components/ui/label/index.js';
	import SafeDelete from '$lib/components/SafeDelete.svelte';
	import { GET_E_A_R_Q, GET_SCORES } from '$lib/utils/ai_evaluations.ts';
	import { Spinner } from '$lib/components/ui/spinner/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.js';
  import ManualCrop from './ManualCrop.svelte';

  if (!data.student_items)
    throw new Error("Student items failed to load");
  if (!data.student_ai_evaluations)
    throw new Error("Student AI evaluations failed to load");

  let isWantsToDelete: boolean = $state(false);
  let studentItems: (StudentAnswer & GetSpecificEvaluationResponse)[] = $derived((() => {
    const evalMap = new Map<number, GetSpecificEvaluationResponse>();
    
    for (const evalItem of data.student_ai_evaluations)
      evalMap.set(evalItem.item_id, evalItem);

    return data.student_items!.map(item => {
      const evalData = evalMap.get(item.item_id)!;
      return {
        ...item,
        answer_id: evalData.answer_id,
        question: evalData.question,
        expected_answer_rubric_questions: evalData.expected_answer_rubric_questions,
        scores: evalData.scores};
    })
  })());

  let isRequestOngoings: Map<number, boolean> = $state(new Map());


  async function reevaluateAnswer(answer_id: number) {
    isRequestOngoings = new Map(isRequestOngoings.set(answer_id, true));
    try {
      const response = await fetch(
            `/api/answers/${answer_id}/reevaluate`,
            {
              method: "PATCH",
              headers: {'Content-Type': 'application/json',},
            }
            );

      switch (response.status) {
        case 200:
          const result = await response.json();
          studentItems = studentItems.map(ans => 
                  ans.answer_id == result.answer_id 
                  ? { ...ans, ai_evaluation: result.ai_evaluation, scores: result.scores }
                  : ans
                );
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

  async function deleteAnswer(item_id: number) {
    console.log(item_id);
    const response = await fetch(`/api/student_answers/${item_id}/${data.student_no}`, { method: "DELETE" });
    switch (response.status) {
      case 204:
        alert(`Deletion of ${item_id} for ${data.student_no} successful.`);
        window.location.reload();
        break;
      default:
        alert(`${response.status} ${response.statusText}`);
    }
  }
</script>


<div class="flex flex-col gap-3">
  <span class="flex flex-row justify-between items-center">
    <span class="flex flex-wrap items-baseline gap-x-4 [&>h5]:opacity-60">
      <h1 class="font-semibold">Test answers</h1>
      <h5>{data.student_no}</h5>
    </span>
    <a class="button-secondary"
        href="/instances/{data.test_id}/papers/{data.student_no}/process">
      <MdiImagePlus class="size-5 mx-2"/>
    </a>
  </span>

  {#if studentItems.length == 0}
    <p>Nothing to see here. <br>If this is a mistake, check your network connection.</p>
  
  {:else}
    <div class="overflow-y-auto space-y-2 flex flex-col items-center">
    {#each studentItems as studentItem}
      {@const isRequestLoading = isRequestOngoings.get(studentItem.answer_id)}
      {@const e_a_r_qs = GET_E_A_R_Q(studentItem)}
      <div class="subcontainer card flex flex-col sm:flex-row gap-x-3 gap-y-1.5">
        <span class="w-full sm:w-1/2 md:w-2/5 lg:w-1/3
                      flex justify-center items-center relative">
          <Label for={studentItem.label}
                  class="absolute top-0 left-0 bg-white px-1.5 text-lg">
            {studentItem.label}
          </Label>
          <span>
            {#if studentItem.image_directory == ""}
              <MdiPaperOff class="size-8 opacity-50" />
            {:else}
              <img class="max-h-70 w-auto mx-auto"
                    src={`${studentItem.image_directory}`}
                    alt={studentItem.label}/>
            {/if}
          </span>
        </span>
        <div class="flex-1 w-full h-full space-y-2">
          <span class="flex flex-row space-x-1 justify-end">
            <SafeDelete toggle={isWantsToDelete}
                        onDelete={() => deleteAnswer(studentItem.item_id)}
                        size={4}
                        />
            <!-- <a href="/instances/{data.test_id}/papers/{data.student_no}/manual?item_id={studentItem.item_id}"
                  class="button-outline">
              <MdiCrop/>
            </a> -->
            <Dialog.Root>
              <Dialog.Trigger class="button-outline">
                <MdiCrop/>
              </Dialog.Trigger>
              <ManualCrop test_id={data.test_id}
                          student_no={data.student_no}
                          item_id={studentItem.item_id}/>
            </Dialog.Root>
            <button class={`${isRequestLoading || studentItem.image_directory == "" ? "opacity-50" : "opacity-100"}
                            button-outline px-0 py-0`}
                    onclick={() => reevaluateAnswer(studentItem.answer_id)}
                    disabled={isRequestLoading || studentItem.image_directory == ""}>
              <IconReevaluate />
            </button>
          </span>
          <div>
            {#each e_a_r_qs as e_a_r_q, index}
            {#if e_a_r_q.length != 0}
              {@const answerScore = GET_SCORES(studentItem)[index]}
              {@const isHasScore = answerScore && answerScore != ""}
              <span class="flex flex-wrap justify-between items-center [&>h5]:opacity-60">
                <h5 class="italic">{e_a_r_q}</h5>
                {#if isRequestLoading}
                  <Spinner class="text-chart-3 size-4" />
                {:else}
                  <h5 class="font-bold">
                    {isHasScore ? answerScore : "—"}
                  </h5>
                {/if}
              </span>
            {/if}
            {/each}
          </div>
        </div>
      </div>
    {/each}
    </div>
  {/if}
</div>

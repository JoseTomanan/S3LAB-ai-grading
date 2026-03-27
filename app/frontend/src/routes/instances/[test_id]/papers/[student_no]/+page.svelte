<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  const { data } = $props();

  import { onDestroy } from 'svelte';
  import { invalidateAll } from '$app/navigation';
  import { api, ApiError } from '$lib/utils/api.ts';
  import toast from 'svelte-5-french-toast';
  import { createPoller } from '$lib/utils/poller.ts';
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
	import Card from '$lib/components/Card.svelte';
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
      }).sort((a, b) => a.label.localeCompare(b.label, undefined, { numeric: true }));
  })());

  let isRequestOngoings: Map<number, boolean> = $state(new Map());
  let pollingItemIds: Set<number> = $state(new Set());
  let cropDialogOpen: Map<number, boolean> = $state(new Map());

  const cropPoller = createPoller(async () => {
    const result = await api<StudentAnswer[]>(
      `${API_URL}/api/student_answers/${data.test_id}/${data.student_no}`
    );
    for (const itemId of pollingItemIds) {
      const answer = result.find(a => a.item_id === itemId);
      if (answer?.is_done_rendering) {
        pollingItemIds = new Set([...pollingItemIds].filter(id => id !== itemId));
      }
    }
    if (pollingItemIds.size === 0) {
      await invalidateAll();
      return true;
    }
    return false;
  }, 5000);

  onDestroy(() => cropPoller.stop());

  function handleCropSubmitted(itemId: number) {
    cropDialogOpen = new Map(cropDialogOpen.set(itemId, false));
    pollingItemIds = new Set(pollingItemIds.add(itemId));
    cropPoller.start();
  }

  async function reevaluateAnswer(answer_id: number) {
    isRequestOngoings = new Map(isRequestOngoings.set(answer_id, true));
    try {
      const result = await api<{ answer_id: number, ai_evaluation: string, scores: string }>(
            `${API_URL}/api/answers/${answer_id}/reevaluate`,
            { method: "PATCH" }
            );
      studentItems = studentItems.map(ans =>
              ans.answer_id == result.answer_id
              ? { ...ans, ai_evaluation: result.ai_evaluation, scores: result.scores }
              : ans
            );
    } catch (e) {
      toast.error(e instanceof ApiError
        ? `${e.status} ${e.statusText}`
        : "Failed to reevaluate answer:\n" + String(e));
    } finally {
      isRequestOngoings = new Map(isRequestOngoings.set(answer_id, false));
    }
  }

  async function deleteAnswer(item_id: number) {
    try {
      await api(`${API_URL}/api/student_answers/${item_id}/${data.student_no}`, { method: "DELETE" });
      toast.success(`Deletion of ${item_id} for ${data.student_no} successful.`);
      await invalidateAll();
    } catch (e) {
      toast.error(e instanceof ApiError
        ? `${e.status} ${e.statusText}`
        : "Failed to delete answer:\n" + String(e));
    }
  }
</script>



<div class="flex flex-col gap-3
            [&>*>img]:object-fit">
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
      {@const isEvalNotError = !studentItem.ai_evaluation.startsWith("_ERROR:")}
      {@const isRequestLoading = isRequestOngoings.get(studentItem.answer_id) || pollingItemIds.has(studentItem.item_id)}
      {@const e_a_r_qs = GET_E_A_R_Q(studentItem)}
      
      <Card class="subcontainer flex flex-col sm:flex-row gap-x-3 gap-y-1.5">
        <span class="flex-1
                      flex justify-center items-center relative">
          <Label for={studentItem.label}
                  class="absolute top-1 -left-1 bg-white px-1.5 text-base shadow-sm">
            {studentItem.label}
          </Label>
          <!-- TODO: Click to open image in a dialog -->
          <span class="w-5/6 sm:w-full">
            {#if studentItem.image_directory == ""}
              <MdiPaperOff class="mx-auto size-8 opacity-50" />
            {:else}
              <!-- FIXME: not working in production (but working in dev somehow??) -->
              <img class="max-h-70 w-auto mx-auto"
                    src={`${API_URL}${studentItem.image_directory}`}
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
            <Dialog.Root
              open={cropDialogOpen.get(studentItem.item_id) ?? false}
              onOpenChange={(v) => { cropDialogOpen = new Map(cropDialogOpen.set(studentItem.item_id, v)); }}>
              <Dialog.Trigger class="button-outline">
                <MdiCrop/>
              </Dialog.Trigger>
              <ManualCrop test_id={data.test_id}
                          student_no={data.student_no}
                          item_id={studentItem.item_id}
                          onCropSubmitted={() => handleCropSubmitted(studentItem.item_id)}/>
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
              <span class="flex flex-wrap justify-between items-center [&>h5]:opacity-60">
                <h5 class="italic">{e_a_r_q}</h5>
                {#if isRequestLoading}
                  <Spinner class="text-chart-3 size-4" />
                {:else}
                  <h4 class="font-semibold">
                    {isEvalNotError
                      ? GET_SCORES(studentItem)[index] || "—"
                      : "⚠️"}
                  </h4>
                {/if}
              </span>
            {/if}
            {/each}
          </div>

        </div>
      </Card>
    {/each}
    </div>
  {/if}
</div>

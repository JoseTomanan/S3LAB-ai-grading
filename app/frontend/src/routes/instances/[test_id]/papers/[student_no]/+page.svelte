<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  const { data } = $props();

  import { onDestroy, onMount } from 'svelte';
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
	import { Skeleton } from '$lib/components/ui/skeleton/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.js';
	import * as Sheet from '$lib/components/ui/sheet/index.ts';
  import { Button, buttonVariants } from '$lib/components/CustomButton/index.ts';
  import { cn } from '$lib/utils.ts';
	import Card from '$lib/components/Card.svelte';
  import ManualCrop from './ManualCrop.svelte';

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
  type CropUiState = {
    cropDialogOpen: boolean;
    cropPending: boolean;
  };
  let cropUiStateByItemId: Map<number, CropUiState> = $state(new Map());
  let zoomLevel: number = $state(1);

  function getCropUiState(itemId: number): CropUiState {
    return cropUiStateByItemId.get(itemId) ?? { cropDialogOpen: false, cropPending: false };
  }

  function updateCropUiState(itemId: number, patch: Partial<CropUiState>) {
    const next = new Map(cropUiStateByItemId);
    next.set(itemId, { ...getCropUiState(itemId), ...patch });
    cropUiStateByItemId = next;
  }

  function setRequestOngoing(answerId: number, isOngoing: boolean) {
    const next = new Map(isRequestOngoings);
    next.set(answerId, isOngoing);
    isRequestOngoings = next;
  }

  const cropPoller = createPoller(async () => {
    const result = await api<StudentAnswer[]>(
      `${API_URL}/api/student_answers/${data.test_id}/${data.student_no}`
    );

    const pendingItemIds = Array.from(cropUiStateByItemId.entries())
      .filter(([, state]) => state.cropPending)
      .map(([itemId]) => itemId);

    for (const itemId of pendingItemIds) {
      const answer = result.find(a => a.item_id === itemId);
      if (answer?.is_done_rendering) {
        updateCropUiState(itemId, { cropPending: false, cropDialogOpen: false });
      }
    }

    const hasPendingCrop = Array.from(cropUiStateByItemId.values()).some(state => state.cropPending);
    if (!hasPendingCrop) {
      await invalidateAll();
      return true;
    }
    return false;
  }, 5000);

  onDestroy(() => cropPoller.stop());

  onMount(() => {
    if (data.evaluation_error)
      toast.error(`Failed to load AI evaluations: ${data.evaluation_error}`);
  });

  function handleCropSubmitted(itemId: number) {
    updateCropUiState(itemId, { cropDialogOpen: false, cropPending: true });
    studentItems = studentItems.map(answer =>
      answer.item_id === itemId
        ? { ...answer, is_done_rendering: false }
        : answer
    );
    cropPoller.start();
  }

  async function reevaluateAnswer(answer_id: number) {
    setRequestOngoing(answer_id, true);
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
        ? (e.detail ? e.detail : `${e.status} ${e.statusText}`)
        : "Failed to reevaluate answer:\n" + String(e));
    } finally {
      setRequestOngoing(answer_id, false);
    }
  }

  async function deleteAnswer(item_id: number) {
    try {
      await api(`${API_URL}/api/student_answers/${item_id}/${data.student_no}`, { method: "DELETE" });
      toast.success(`Deletion of ${item_id} for ${data.student_no} successful.`);
      await invalidateAll();
    } catch (e) {
      toast.error(e instanceof ApiError
        ? (e.detail ? e.detail : `${e.status} ${e.statusText}`)
        : "Failed to delete answer:\n" + String(e));
    }
  }
</script>



<div class="p-4 flex flex-col gap-3">
  <div class="flex items-center justify-between">
    <div>
      <span class="font-heading text-[20px] font-extrabold text-foreground tracking-[-0.04em]">Test Answers</span>
      <span class="ml-2 text-[12px] font-mono px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
        {data.student_no}
      </span>
    </div>
    <a
      href="/instances/{data.test_id}/papers/{data.student_no}/process"
      class="size-9 rounded-full bg-primary flex items-center justify-center text-white hover:opacity-90 transition-opacity"
      title="Upload image"
    >
      <MdiImagePlus class="size-[18px]" />
    </a>
  </div>

  {#if studentItems.length == 0}
    <p>Nothing to see here. <br>If this is a mistake, check your network connection.</p>
  
  {:else}
    <div class="scroll-shadows overflow-y-auto space-y-2 flex flex-col items-center">
    {#each studentItems as studentItem}
      {@const isEvalNotError = !studentItem.ai_evaluation.startsWith("_ERROR:")}
      {@const cropUiState = getCropUiState(studentItem.item_id)}
      {@const isRequestLoading = isRequestOngoings.get(studentItem.answer_id)
                || cropUiState.cropPending
                || (studentItem.image_directory !== "" && !studentItem.is_done_rendering)}
      {@const e_a_r_qs = GET_E_A_R_Q(studentItem)}
      
      <Card class="subcontainer flex flex-col sm:flex-row gap-x-3 gap-y-1.5">
        <span class="flex-1 flex justify-center items-center relative">
          <Label for={studentItem.label}
                  class="absolute top-1 -left-1 bg-white px-1.5 text-base shadow-sm">
            {studentItem.label}
          </Label>
          <span class="w-5/6 sm:w-full">
            {#if isRequestLoading && studentItem.image_directory == ""}
              <Skeleton class="w-full h-32 rounded-none grayscale" />
            {:else if studentItem.image_directory == ""}
              <MdiPaperOff class="mx-auto size-8 opacity-50" />
            {:else}
              <Dialog.Root onOpenChange={(v) => { if (!v) zoomLevel = 1; }}>
                <Dialog.Trigger class="w-full cursor-zoom-in">
                  <!-- FIXME: Not working in production. Remove this when validated that it's good -->
                  <enhanced:img class="max-h-70 w-auto mx-auto"
                        src={`${API_URL}${studentItem.image_directory}`}
                        alt={studentItem.label}/>
                </Dialog.Trigger>
                <Dialog.Content class="sm:max-w-[90vw] p-2 gap-1">
                  <Dialog.Title class="sr-only">{studentItem.label}</Dialog.Title>
                  <div class="flex max-h-[85vh] justify-center overflow-auto"
                       onwheel={(e) => {
                         e.preventDefault();
                         zoomLevel = Math.min(5, Math.max(0.5, zoomLevel - e.deltaY * 0.001));
                       }}>
                    <enhanced:img style={`zoom: ${zoomLevel};`}
                         class="block"
                         src={`${API_URL}${studentItem.image_directory}`}
                         alt={studentItem.label}/>
                  </div>
                </Dialog.Content>
              </Dialog.Root>
            {/if}
          </span>
        </span>
        
        <div class="flex-1 w-full h-full space-y-2">
          <span class="flex flex-row space-x-1 justify-end">
            <SafeDelete onDelete={() => deleteAnswer(studentItem.item_id)}/>
            <Sheet.Root open={cropUiState.cropDialogOpen}
                          onOpenChange={(v) => { updateCropUiState(studentItem.item_id, { cropDialogOpen: v }); }}>
              <Sheet.Trigger class={buttonVariants({ variant: 'outline' })}>
                <MdiCrop/>
              </Sheet.Trigger>
              <ManualCrop test_id={data.test_id}
                          student_no={data.student_no}
                          item_id={studentItem.item_id}
                          onCropSubmitted={() => handleCropSubmitted(studentItem.item_id)}/>
            </Sheet.Root>
            <Button variant="outline"
                    onclick={() => reevaluateAnswer(studentItem.answer_id)}
                    disabled={isRequestLoading || studentItem.image_directory == ""}>
              <IconReevaluate />
            </Button>
          </span>
          
          <div>
            {#each e_a_r_qs as e_a_r_q, index}
            {#if e_a_r_q.length != 0}
              <span class="flex flex-row justify-between items-baseline-last gap-x-1">
                <p class="italic flex-1 text-muted-foreground">
                  {e_a_r_q}
                </p>
                {#if isRequestLoading}
                  <Spinner class="text-chart-3 size-4" />
                {:else}
                  <p class="font-semibold text-foreground/80">
                    {isEvalNotError
                      ? GET_SCORES(studentItem)[index] || "—"
                      : "⚠️"}
                  </p>
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

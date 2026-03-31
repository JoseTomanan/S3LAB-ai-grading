<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  const { data } = $props();

  import { onMount, onDestroy, untrack } from "svelte";

  import IconStatusDone from '~icons/mdi/paper-check-outline';
  import IconStatusUndone from '~icons/mdi/paper-outline';
  import IconPlus from '~icons/mdi/plus';

  import type { GetEvaluationsResponse } from '$lib/index.ts';
	import { Spinner } from '$lib/components/ui/spinner/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import { api } from '$lib/utils/api.ts';
  import { createPoller } from '$lib/utils/poller.ts';

	import AddNewStudent from '@/routes/sections/[section_id]/AddNewStudent.svelte';

  let isAddStudentDialogOpen: boolean = $state(false);
  let perStudentStatuses: GetEvaluationsResponse[] = $state(untrack(() => data.statuses));
  $effect(() => {
    perStudentStatuses = data.statuses;
  });
  let isPolling = $derived(perStudentStatuses.some(s => s.has_any_answer && !s.is_done_rendering));

  const poller = createPoller(async () => {
    const result = await api<{ statuses: GetEvaluationsResponse[] }>(
      `${API_URL}/api/student_answers/${data.test_id}/statuses`
    );
    perStudentStatuses = result.statuses;
    return perStudentStatuses.length > 0 && perStudentStatuses.every(s => s.is_done_rendering);
  }, 5000);

  onMount(() => {
    if (isPolling) {
      poller.start();
    }
  });

  onDestroy(() => poller.stop());
</script>



<div class="flex flex-col gap-y-3 overflow-visible items-center">
  <h1 class="text-left font-semibold w-full flex justify-between items-center">
    Test papers
    <Dialog.Root bind:open={isAddStudentDialogOpen}>
      <Dialog.Trigger class="button-secondary">
        <IconPlus class="size-5"/>
      </Dialog.Trigger>
      <AddNewStudent bind:isAddDialogOpen={isAddStudentDialogOpen}
                      section_id={data.test_instance?.section_id}/>
    </Dialog.Root>
  </h1>
  
  {#if perStudentStatuses.length == 0}
    <p>Nothing to see here. <br>If this is a mistake, check your network connection.</p>

  {:else}
    {#each perStudentStatuses as testPaper}
    <a href={`/instances/${data.test_id}/papers/${testPaper.student_no}`}
        class="flex-1 button-outline subcontainer
                flex flex-row justify-between items-center">
      <h4 class="truncate space-x-1">
        <span class="tracking-tighter">{testPaper.student_no}</span>
        <span class="font-normal">{testPaper.name}</span>
      </h4>
      <h5 class="flex flex-row gap-x-1.5 items-center tracking-tighter
                  *:size-5">
        {testPaper.total_score}
        {#if testPaper.has_any_answer && !testPaper.is_done_rendering}
          <Spinner class="text-chart-3"/>
        {:else if testPaper.is_done_rendering}
          <IconStatusDone/>
        {:else}
          <IconStatusUndone/>
        {/if}
      </h5>
    </a>
    {/each}
  {/if}
</div>

<script lang="ts">
  const { data } = $props();

  import { onMount, onDestroy } from "svelte";

  import MdiPaperAlertOutline from '~icons/mdi/paper-alert-outline';
  import MdiPaperCheckOutline from '~icons/mdi/paper-check-outline';

  import type { GetEvaluationsResponse } from '$lib/index.ts';
	import { Spinner } from '$lib/components/ui/spinner/index.ts';
  import { api } from '$lib/utils/api.ts';
  import { createPoller } from '$lib/utils/poller.ts';

  let perStudentStatuses: GetEvaluationsResponse[] = $state(data.statuses);
  $effect(() => {
    perStudentStatuses = data.statuses;
  });
  let isPolling = $derived(perStudentStatuses.some(s => !s.is_done_rendering));

  const poller = createPoller(async () => {
    const result = await api<{ statuses: GetEvaluationsResponse[] }>(
      `/api/student_answers/${data.test_id}/statuses`
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
  <h1 class="text-left font-semibold w-full flex justify-between items-center gap-x-2">
    Test papers
    <!-- Removed for now because not working properly. FIXME: fix polling stuffs -->
    <!--
    {#if isPolling}
      <Spinner class="size-4 text-muted-foreground/80"/>
    {/if}
    -->
  </h1>
  {#if perStudentStatuses.length == 0}
    <p>Nothing to see here. <br>If this is a mistake, check your network connection.</p>

  {:else}
    {#each perStudentStatuses as testPaper}
    <a href={`/instances/${data.test_id}/papers/${testPaper.student_no}`}
        class="flex-1 button-outline subcontainer
                flex flex-row justify-between items-center">
      <h4>{testPaper.name}</h4>
      <h5 class="flex flex-row gap-x-1.5 items-center">
        {testPaper.total_score}
        {#if testPaper.is_done_rendering}
          <MdiPaperCheckOutline class="size-5"/>
        {:else}
          <MdiPaperAlertOutline class="size-5"/>
        {/if}
      </h5>
    </a>
    {/each}
  {/if}
</div>

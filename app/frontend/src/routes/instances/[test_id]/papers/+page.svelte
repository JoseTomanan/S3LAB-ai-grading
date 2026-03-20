<script lang="ts">
  const { data } = $props();

    import { onMount } from "svelte";

  import MdiPaperAlertOutline from '~icons/mdi/paper-alert-outline';
  import MdiPaperCheckOutline from '~icons/mdi/paper-check-outline';

  import type { GetEvaluationsResponse } from '$lib/index.ts';
	import { Skeleton } from '$lib/components/ui/skeleton/index.ts';
	import { Spinner } from '$lib/components/ui/spinner/index.ts';

  let isPageLoading: boolean = $state(true);
  let perStudentStatuses: GetEvaluationsResponse[] = $state([]);
  let pollInterval: ReturnType<typeof setInterval> | null = $state(null);

  async function fetchStatuses() {
    const response = await fetch(
          `/api/student_answers/${data.test_id}/statuses`,
          {
            method: "GET",
            headers: {'Content-Type': 'application/json',},
          }
          );

    if (response.ok) {
      const result = await response.json();
      perStudentStatuses = result.statuses;

      if (perStudentStatuses.length > 0 && perStudentStatuses.every(s => s.is_done_rendering)) {
        if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
      }
    }
  }

  onMount(() => {
    (async () => {
      try {
        await fetchStatuses();
      } catch(e) {
        alert("Failed to fetch test paper statuses:\nERROR: "+e);
      } finally {
        isPageLoading = false;
      }

      if (perStudentStatuses.some(s => !s.is_done_rendering)) {
        pollInterval = setInterval(fetchStatuses, 5000);
      }
    })();

    return () => { if (pollInterval) clearInterval(pollInterval); };
  });
</script>


<div class="flex flex-col gap-y-3 overflow-visible items-center">
  <h1 class="text-left font-semibold w-full flex justify-between items-center gap-x-2">
    Test papers
    {#if pollInterval}
      <Spinner class="size-4 text-muted-foreground/80"/>
    {/if}
  </h1>
  {#if isPageLoading}
    {#each { length: 3 } as _}
      <Skeleton class="subcontainer h-10 w-full grayscale"/>
    {/each}

  {:else if perStudentStatuses.length == 0}
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

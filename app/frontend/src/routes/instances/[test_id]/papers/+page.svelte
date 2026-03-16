<script lang="ts">
  const { data } = $props();
  
  import { API_BASE_URL } from '$lib/constants.ts';
  import { onMount } from "svelte";
  
  import MdiPaperAlertOutline from '~icons/mdi/paper-alert-outline';
  import MdiPaperCheckOutline from '~icons/mdi/paper-check-outline';

  import type { GetEvaluationsResponse } from '$lib/index.ts';
	import { Skeleton } from '$lib/components/ui/skeleton/index.ts';

  let isPageLoading: boolean = $state(true);
  let perStudentStatuses: GetEvaluationsResponse[] = $state([]);

  onMount(async () => {
    try {
      const response = await fetch(
            `${API_BASE_URL}/api/student_answers/${data.test_id}/statuses`,
            {
              method: "GET",
              headers: {'Content-Type': 'application/json',},
            }
            );

      switch (response.status) {
        case 200:
          const result = await response.json();
          perStudentStatuses = result.statuses;
          break;
        default:
          alert(`${response.status} ${response.statusText}`);
      }
    } catch(e) {
      alert("Failed to fetch test paper statuses:\nERROR: "+e);
    } finally {
      isPageLoading = false;
    }
  });
</script>


<div class="flex flex-col gap-y-3 overflow-visible items-center">
  <h1 class="text-left font-semibold w-full">Test papers</h1>
  {#if isPageLoading}
    {#each { length: 3 } as _}
      <Skeleton class="h-10 w-full grayscale-100
                    md:w-5/6 lg:w-3/4"/>
    {/each}
  
  {:else if perStudentStatuses.length == 0}
    <p>Nothing to see here. <br>If this is a mistake, check your network connection.</p>
  
  {:else}
    {#each perStudentStatuses as testPaper}
    <a href={`/instances/${data.test_id}/papers/${testPaper.student_no}`}
        class="flex-1 button-outline
                w-full md:w-5/6 lg:w-3/4
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

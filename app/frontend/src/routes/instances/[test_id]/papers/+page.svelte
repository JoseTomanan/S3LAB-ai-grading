<script lang="ts">
  const { data } = $props();
  
  import { API_BASE_URL } from '$lib/constants.ts';
  import { onMount } from "svelte";
  
  import MdiPaperAlertOutline from '~icons/mdi/paper-alert-outline';
  import MdiPaperCheckOutline from '~icons/mdi/paper-check-outline';

  import * as Dialog from '$lib/components/ui/dialog/index.ts';

  import GetAIEvaluation from './GetAIEvaluation.svelte';
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


<div class="space-y-3 overflow-visible">
  <h1 class="text-left font-semibold">Test papers</h1>
  {#if isPageLoading}
    {#each { length: 3 } as _}
      <Skeleton class="h-10 w-full grayscale-100"/>
    {/each}
  {:else if perStudentStatuses.length == 0}
    <p>Nothing to see here. <br>If this is a mistake, check your network connection.</p>
  {:else}
    {#each perStudentStatuses as testPaper}
      <span class="flex flex-row items-center justify-between gap-2">
        <a href={`/instances/${data.test_id}/papers/${testPaper.student_no}`}
            class="flex-1 button-outline
                    flex flex-row justify-between items-center">
          <h4>{testPaper.name}</h4>
          {#if testPaper.is_done_rendering}
            <MdiPaperCheckOutline class="size-5"/>
          {:else}
            <MdiPaperAlertOutline class="size-5"/>
          {/if}
        </a>
        <GetAIEvaluation test_id={data.test_id}
                          student_no={testPaper.student_no}/>
      </span>
    {/each}
  {/if}
</div>

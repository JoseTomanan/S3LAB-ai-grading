<script lang="ts">
  import { API_BASE_URL } from '$lib/constants.ts';
  import { onMount } from 'svelte';

  import MdiCheckboxBlankOutline from '~icons/mdi/checkbox-blank-outline';
  import MdiCheckboxMarkedOutline from '~icons/mdi/checkbox-marked-outline';
  import MdiPaperAddOutline from '~icons/mdi/paper-add-outline';
  import MdiPeopleOutline from '~icons/mdi/people-outline';
  import IconHome from '~icons/mdi/home-outline';
  
  import type { TestInstance } from '$lib/index.ts';
  
  import Pagination from '$lib/components/Pagination.svelte';
  import * as Dialog from "$lib/components/ui/dialog/index.js";
  import AddTestInstance from './AddTestInstance.svelte';
	import { Skeleton } from '$lib/components/ui/skeleton/index.ts';
  
  let isPageLoading: boolean = true;
  let instances: TestInstance[] = [];
  let paginationValues: TestInstance[];

  onMount(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/test_instances`);

      switch (response.status) {
        case 200:
          const result = await response.json();
          instances = result ? result : [];
          break;
        default:
          alert(`${response.status} ${response.statusText}`);
      }
    } catch (e) {
      alert("Failed to fetch test instances:\n"+e);
    } finally {
      isPageLoading = false;
    }
  });
</script>


<div class="container">
  <span class="flex flex-row items-center justify-between mb-4">
    <Dialog.Root>
      <Dialog.Trigger class="button-primary">
        <MdiPaperAddOutline class="size-6"/>
      </Dialog.Trigger>
      <AddTestInstance />
    </Dialog.Root>
    <h1>Test instances</h1>
    <span class="flex flex-row gap-x-2">
      <a href="/">
        <IconHome class="size-6"/>
      </a>
      <a href="/sections">
        <MdiPeopleOutline class="size-6"/>
      </a>
    </span>
  </span>
  <div class="flex flex-col gap-3">
    {#if isPageLoading}
      {#each { length: 2 } as _}
        <Skeleton class="h-16 w-full grayscale-100"/>
      {/each}
    {:else if instances.length == 0}
      <p>Nothing to see here. <br>Check your network connection, or add a new instance.</p>
    {:else}
      {#each paginationValues as instance}
        <a href={`/instances/${instance.test_id}/items`}
            class="card button-outline">
          <span class="flex flex-row items-center gap-1">
            {#if instance.is_done_rendering}
              <MdiCheckboxMarkedOutline/>
            {:else}
              <MdiCheckboxBlankOutline/>
            {/if}
            <h3>{ instance.name }</h3>
          </span>
          <h5>{ instance.test_id.split("_")[0] } &middot; { new Date(instance.date).toLocaleDateString() }</h5>
        </a>
      {/each}
    {/if}
  </div>
  <Pagination rows={instances}
              perPage={6}
              bind:trimmedRows={paginationValues} />
</div>

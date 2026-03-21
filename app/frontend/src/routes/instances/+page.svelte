<script lang="ts">
  const { data } = $props();

  import MdiCheckboxBlankOutline from '~icons/mdi/checkbox-blank-outline';
  import MdiCheckboxMarkedOutline from '~icons/mdi/checkbox-marked-outline';
  import MdiPaperAddOutline from '~icons/mdi/paper-add-outline';
  import IconHome from '~icons/mdi/home-outline';
  import IconArrow from '~icons/mdi/arrow-up';

  import type { TestInstance } from '$lib/index.ts';

  import Pagination from '$lib/components/Pagination.svelte';
  import * as Dialog from "$lib/components/ui/dialog/index.js";
  import AddTestInstance from './AddTestInstance.svelte';

  let instances: TestInstance[] = $derived(data.instances);
  let paginationValues: TestInstance[] = $state([]);
</script>


<div class="container">
  <span class="flex flex-row items-center justify-between mb-4">
    <span class="flex flex-row gap-x-2">
      <a href="/">
        <IconHome class="size-8"/>
      </a>
    </span>
    <h1>Test instances</h1>
    <Dialog.Root>
      <Dialog.Trigger class="button-primary">
        <MdiPaperAddOutline class="size-6"/>
      </Dialog.Trigger>
      <AddTestInstance />
    </Dialog.Root>
  </span>
  <div class="flex flex-col gap-3 relative">
    <a href="/sections" class="w-fit text-base opacity-60 hover:underline">
      Go to sections ↗
    </a>
    {#if instances.length == 0}
      <div class="absolute top-0 right-0
                  flex flex-col items-end text-right mt-2 opacity-60">
        <IconArrow class="size-10"/>
        <p>No test instances yet.<br>Tap here to add one!</p>
      </div>
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

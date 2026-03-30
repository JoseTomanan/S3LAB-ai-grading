<script lang="ts">
  const { data } = $props();

  import MdiPaperAddOutline from '~icons/mdi/paper-add';
  import IconHome from '~icons/mdi/home-outline';
  import IconSections from '~icons/mdi/people';

  import type { TestInstance } from '$lib/index.ts';

  import Pagination from '$lib/components/Pagination.svelte';
  import * as Dialog from "$lib/components/ui/dialog/index.js";
  import AddTestInstance from './AddTestInstance.svelte';
	import Card from '$lib/components/Card.svelte';

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
    <a href="/sections">
      <IconSections class="size-8 text-primary saturate-200 brightness-60"/>
    </a>
  </span>

  <div class="flex flex-col gap-3 relative">
    <Dialog.Root>
      <Dialog.Trigger class="button-outline flex flex-row gap-x-2 items-center justify-center font-semibold
                              *:opacity-90 *:font-semibold">
        <MdiPaperAddOutline class="size-5"/>
        <b>Add new test instance</b>
      </Dialog.Trigger>
      <AddTestInstance />
    </Dialog.Root>
    
    {#if instances.length == 0}
      <div class="absolute top-0 right-0
                  flex flex-col items-center text-center mt-2 opacity-60">
        <p>No test instances yet. Tap above to add one!</p>
      </div>
    
    {:else}
      {#each paginationValues as instance}
        <Card href="/instances/{instance.test_id}/items"
              class="button-outline">
          <h3 class="flex flex-row items-center gap-1">
            {instance.name} &middot; {instance.test_id.split("_")[0]}
          </h3>
          <h5 class="font-normal">
            Created {new Date(instance.date).toLocaleDateString()}
          </h5>
        </Card>
      {/each}
    {/if}
  </div>
  <Pagination rows={instances}
              perPage={6}
              bind:trimmedRows={paginationValues} />
</div>

<script lang="ts">
  const { data } = $props();

  import MdiPaperAddOutline from '~icons/mdi/paper-add';
  import IconHome from '~icons/mdi/home';
  import IconSections from '~icons/mdi/people-outline';
  import IconDone from '~icons/mdi/checkbox-marked-circle';
  import IconNotDone from '~icons/mdi/checkbox-blank-circle';

  import type { TestInstance } from '$lib/index.ts';
  import { buttonVariants } from '$lib/components/CustomButton/index.ts';
  import { cn } from '$lib/utils.ts';

  import Pagination from '$lib/components/Pagination.svelte';
  import * as Dialog from "$lib/components/ui/dialog/index.js";
  import AddTestInstance from './AddTestInstance.svelte';
	import { Separator } from '$lib/components/ui/separator/index.ts';

  let instances: TestInstance[] = $derived(data.instances);
  let paginationValues: TestInstance[] = $state([]);
</script>



<nav class="w-full flex flex-row items-center justify-between mb-4
            px-4 pt-6">
  <span class="flex flex-row gap-x-2">
    <a href="/" class={buttonVariants({ variant: 'floating' })}>
      <IconHome class="size-8"/>
    </a>
  </span>
  <h1>Test instances</h1>
  <a href="/sections" class={buttonVariants({ variant: 'floating' })}>
    <IconSections class="size-8"/>
  </a>
</nav>
<span class="-my-5"></span>

<div class="container">

  <div class="flex flex-col gap-3 relative">
    <Dialog.Root>
      <Dialog.Trigger class={cn(buttonVariants({ variant: 'outline' }), 'flex flex-row gap-x-2 items-center justify-center text-base *:font-semibold *:opacity-90')}>
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
      <a href="/instances/{instance.test_id}/items"
          class={cn(buttonVariants({ variant: 'outline' }), 'justify-between px-3 py-1.5 *:text-base')}>
        <span>
          {instance.name} &middot; {instance.test_id.split("_")[0]}
        </span>
        <h5 class="font-normal flex flex-row items-center gap-x-1.5 ml-0.5
                    *:opacity-60">
          {new Date(instance.date).toLocaleDateString()}
          {#if instance.is_done_rendering}
            <IconDone class="size-3"/>
          {:else}
            <IconNotDone class="size-3"/>
          {/if}
        </h5>
      </a>
      {/each}
    {/if}
  </div>
  <Pagination rows={instances}
              perPage={6}
              bind:trimmedRows={paginationValues} />
</div>

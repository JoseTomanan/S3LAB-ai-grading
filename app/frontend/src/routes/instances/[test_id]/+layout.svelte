<script lang="ts">
  let { data, children }: {data: LayoutData, children: Snippet} = $props();

  import { navigating, page } from '$app/state';
  import type { LayoutData } from './$types.d.ts';
  import type { Snippet } from 'svelte';
  import { setContext, untrack } from 'svelte';

  import BulkUpload from './BulkUpload.svelte';
  import ExportSheets from './ExportSheets.svelte';

  import IconInstancesList from '~icons/mdi/format-list-bulleted';
  import IconBack from '~icons/mdi/chevron-left';
  import IconTable from '~icons/mdi/table';
  import IconUpload from '~icons/mdi/tray-upload';

  import type { TestInstance, TestItemsContext } from '$lib/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.ts';
	import * as Sheet from '$lib/components/ui/sheet/index.ts';
  import { Button, buttonVariants } from '$lib/components/ui/button/index.ts';
  import { cn } from '$lib/utils.ts';
	import { Separator } from '$lib/components/ui/separator/index.ts';
	import { Skeleton } from '$lib/components/ui/skeleton/index.ts';
	import { goto, invalidateAll } from '$app/navigation';

  const isRouteItems = $derived(page.route.id!.includes('/items'));
  const isRoutePapers = $derived(page.route.id!.includes('/papers'));

  let activeTestInstance: TestInstance = $derived(data.test_instance!);
  let isBulkUploadOpen: boolean = $state(false);

  let testItemsContext: TestItemsContext = $state({items: untrack(() => data.test_items!), isLoading: false});
  $effect(() => {
    testItemsContext.items = data.test_items!;
  });
  setContext("testItemsContext", testItemsContext);
</script>



<!-- <div> -->
  <nav class="bg-sidebar text-sidebar-foreground p-4 pt-6 shadow shadow-sidebar-border space-y-2.5">
    <span class="flex flex-row items-center justify-between">
      <Button variant="floating"
              onclick={() => goto("..")}>
        <IconBack class="size-8"/>
      </Button>
      <h1>{ activeTestInstance.name }</h1>
      <a href="/instances" class={buttonVariants({ variant: 'floating' })}>
        <IconInstancesList class="size-8"/>
      </a>
    </span>
    
    <Separator/>
    <div class="flex flex-col sm:flex-row gap-x-4 gap-y-1
                *:flex *:justify-between
                [&>*>h4]:font-normal [&>*>h4]:text-foreground/85 [&>*>h4]:leading-none [&>*>h4]:whitespace-nowrap">
      <span class="flex-3/5 tracking-tight">
        <h4 class="italic">{activeTestInstance.test_id}</h4>
        <h4>
          Rendered?
          {activeTestInstance.is_done_rendering 
            ? "☑️" 
            : "✖️" }
        </h4>
      </span>
      <span class="flex-2/5">
        <h4>
          Created
          {activeTestInstance.date
            ? new Date(activeTestInstance.date).toLocaleDateString()
            : "" }
        </h4>
        <h4>SectionID: { activeTestInstance.section_id }</h4>
      </span>
    </div>
    
    <div class="flex items-center justify-between underline-offset-3 *:space-x-1 ">
      <span>
        <a class={cn(buttonVariants(), 'ring-chart-3', isRouteItems ? 'ring-2' : '')}
            href="/instances/{data.test_id}/items">
          Items
        </a>
        <a class={cn(buttonVariants(), 'ring-chart-3', isRoutePapers ? 'ring-2' : '')}
            href="/instances/{data.test_id}/papers">
          Papers
        </a>
      </span>
      
      <span class="flex flex-row gap-x-1">
        <Dialog.Root>
          <Dialog.Trigger class={buttonVariants()}
                          title={"Export sheets"}>
            <IconTable class="size-5" />
          </Dialog.Trigger>
          <ExportSheets test_id={data.test_id}/>
        </Dialog.Root>
        <Sheet.Root bind:open={isBulkUploadOpen}>
          <Sheet.Trigger class={buttonVariants()}
                          title={"Bulk upload"}>
            <IconUpload class="size-5"/>
          </Sheet.Trigger>
          <BulkUpload test_instance={activeTestInstance}
                      defaultNumBoxes={data.test_items?.length ?? 2}
                      onuploadcomplete={invalidateAll}
                      open={isBulkUploadOpen}
                  />
        </Sheet.Root>
      </span>
    </div>
  </nav>
  
  <div class="container -mt-4">
    {#if navigating.to}
      <Skeleton class="grayscale-50 w-full h-64 rounded-none"/>
    {:else}
      {@render children()}
    {/if}
  </div>
<!-- </div> -->

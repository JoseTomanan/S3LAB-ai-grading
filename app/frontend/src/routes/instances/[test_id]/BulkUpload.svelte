<script lang="ts">
  import { API_BASE_URL } from '$lib/constants.ts';

  const { test_id } = $props();

  import IconCheck from "~icons/mdi/check";
  import IconSend from "~icons/mdi/send";
  import IconPerson from "~icons/mdi/person";

  import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import { Button } from '$lib/components/ui/button/index.ts';
	import { Input } from '$lib/components/ui/input/index.ts';
	import { Spinner } from '$lib/components/ui/spinner/index.ts';
	import BulkUploadRename from './BulkUploadRename.svelte';

  let files: File[] = $state([]);
  let previews: {name: string, url: string}[] = $state([]);
  let isOperationStarted: boolean = $state(false);

  // Map<index: number, requestStatus: boolean>
  let renderStatuses: Map<number, boolean> = $state(new Map());

  $effect(() => {
    console.log(renderStatuses);
  });

  function handleFiles(e: Event) {
    const input = e.target as HTMLInputElement;
    if (!input.files)
      return;

    const selected = Array.from(input.files);
    files = selected;
    previews = selected.map(f => ({
      name: f.name,
      url: URL.createObjectURL(f)
    }));

    const newStatuses = new Map<number, boolean>();
    previews.forEach((_, idx) => newStatuses.set(idx, false));
    renderStatuses = newStatuses;
  }

  async function bulkUpload() {
    isOperationStarted = true;
    console.log(renderStatuses);

    let idx = 0;
    for (const [key, value] of renderStatuses) {
      console.log(key);
      await new Promise(resolve => setTimeout(resolve, 1000));
      renderStatuses = new Map(renderStatuses.set(key, true));
      idx++;
    }
  }
</script>


<Dialog.Content class="w-full *:min-w-0">
  <Input type="file"
          multiple
          accept="image/*"
          class="w-11/12"
          onchange={handleFiles}
          disabled={isOperationStarted}
      />
  {#if files.length == 0}
    <Dialog.Description>
      Make sure files are named with student number, e.g., "202011111.jpeg".
    </Dialog.Description>
  
  {:else if !isOperationStarted}
    <div class="space-y-1">
      <Dialog.Header class="flex flex-row justify-between">
        <span class="block">
          <Dialog.Title>
            Image preview
          </Dialog.Title>
          <Dialog.Description>
            Uploaded {files.length} images
          </Dialog.Description>
        </span>
        <Button variant="secondary"
                  onclick={bulkUpload}>
          <IconSend class="size-4" />
        </Button>
      </Dialog.Header>
      <div class="flex flex-row items-center overflow-x-auto gap-x-1
              -mx-5 px-5 pt-1 pb-2 shadow-sm
              border-b-2 border-t-2 border-border"
           style="scrollbar-gutter: stable; overflow-y: hidden;">
        {#each previews as p}
          {@const supposedId = p.name.substring(0, p.name.lastIndexOf('.'))}
          {@const fileExtension = p.name.substring(p.name.lastIndexOf('.'))}
          <div class="relative flex flex-col justify-end min-w-full sm:min-w-3/4 h-auto gap-y-0.5">
            <img src={p.url}
                  class="aspect-auto block"
                  alt={p.name} />
            <span class="absolute bottom-0 left-0 w-full flex flex-row pl-2">
              <Dialog.Root>
                <Dialog.Trigger class="max-w-full flex flex-row gap-1 items-center truncate px-1.5 mb-1 bg-white/80 backdrop-blur-md 
                                    hover:underline cursor-pointer">
                  <IconPerson class="size-3"/>
                  <h4 class="text-left w-fit truncate">
                    {supposedId}
                  </h4>
                </Dialog.Trigger>
                <BulkUploadRename
                      filename={supposedId}
                      onchange={(studentNo: string) => p.name = studentNo + fileExtension} />
              </Dialog.Root>
            </span>
          </div>
        {/each}
      </div>
    </div>
  
  {:else}
    <div class="flex flex-col space-y-1">
      {#each previews as p, idx}
        <span class="flex flex-row justify-start items-center gap-x-1.5">
          <span class="mr-1">
            {#if renderStatuses.get(idx)}
              <IconCheck class="size-5"/>
            {:else}
              <Spinner class="size-5"/>
            {/if}
          </span>
          <h4 class="truncate w-fit">{p.name}</h4>
        </span>
      {/each}
    </div>
  {/if}
</Dialog.Content>

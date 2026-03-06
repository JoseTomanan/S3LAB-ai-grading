<script lang="ts">
  import { API_BASE_URL } from '$lib/constants.ts';

  const { test_id } = $props();

  import IconSend from "~icons/mdi/send";
  import IconPerson from "~icons/mdi/person";

  import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import { Button } from '$lib/components/ui/button/index.ts';
	import { Input } from '$lib/components/ui/input/index.ts';
	import BulkUploadRename from '$lib/components/BulkUploadRename.svelte';

  let files: File[] = $state([]);
  let previews: {name: string, url: string}[] = $state([]);

  $effect(() => {
    // TODO: filename is according to inputted student number + file extension
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
  }

  async function bulkUpload() {
    // TODO: function
  }
</script>


<Dialog.Content class="w-full *:min-w-0">
  <Input type="file"
          multiple
          accept="image/*"
          class="w-11/12"
          onchange={handleFiles}
      />
  {#if files.length == 0}
    <Dialog.Description>
      Make sure files are named with student number, e.g., "202011111.jpeg".
    </Dialog.Description>
  {:else}
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
        <Button variant="secondary">
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
                <BulkUploadRename filename={p.name} />
              </Dialog.Root>
            </span>
          </div>
        {/each}
      </div>
    </div>
  {/if}
</Dialog.Content>

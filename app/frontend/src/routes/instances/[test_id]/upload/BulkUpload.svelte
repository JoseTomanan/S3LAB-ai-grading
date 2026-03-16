<script lang="ts">
  	import { onMount } from 'svelte';

  const { test_id, section_id } = $props();

  const GET_NAME_ONLY = (s: string) => s.substring(0, s.lastIndexOf('.'));
  const GET_EXTENSION_ONLY = (s: string) => s.substring(s.lastIndexOf('.'));
  const IS_IN_STUDENTS = (x: string) => students.some(s => s.student_no == x)

  import IconCheck from "~icons/mdi/check";
  import IconExclamation from "~icons/mdi/exclamation-thick";
  import IconNotFound from "~icons/mdi/account-question-outline";
  import IconSend from "~icons/mdi/send";
  import IconPerson from "~icons/mdi/person";

  import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import { Button } from '$lib/components/ui/button/index.ts';
	import { Input } from '$lib/components/ui/input/index.ts';
	import { Spinner } from '$lib/components/ui/spinner/index.ts';
	import BulkUploadRename from './BulkUploadRename.svelte';
	import type { FileRecord, Student } from '$lib/index.ts';
  
  let isOperationStarted: boolean = $state(false);
  let formFiles: FileList | undefined = $state();
  let formFileRecords: FileRecord[] = $state([]);
  let students: Student[] = $state([]);


  onMount(async () => {
    try {
      const response = await fetch(`/api/sections/${section_id}`);
      const results = await response.json();
      students = results;
    } catch (e) {
      console.log("Failed to fetch students for this section.");
    }
  });


  function handleFiles() {
    if (!formFiles || formFiles.length == 0)
      return;

    formFileRecords.forEach(r => URL.revokeObjectURL(r.url));
    formFileRecords = Array.from(formFiles).map(f => ({
      tempId: crypto.randomUUID(),
      file: f,
      name: f.name,
      url: URL.createObjectURL(f),
      statusCode: -1
    }));
  }

  async function bulkUpload() {
    isOperationStarted = true;
    try {
      for (const p of formFileRecords) {
        console.log(p);
        const activeStudentNo = GET_NAME_ONLY(p.name);

        if (!IS_IN_STUDENTS(activeStudentNo)) {
          p.statusCode = 404;
          continue;
        }
        
        (async () => {
          const formData = new FormData();
          formData.append('file', p.file);
          const response = await fetch(
                `/api/student_answers/${test_id}/${activeStudentNo}/image_preprocess`,
                { method: "POST", body: formData, }
              );
          console.log(`${p.name}: ${response.status}`);
          p.statusCode = response.status;
        })();
      }
    } catch (e) {
      console.log("Bulk upload operation failed:\n"+e);
    }
  }
</script>


<Dialog.Content class="w-full *:min-w-0">
  <Input type="file"
          multiple
          accept="image/*"
          bind:files={formFiles}
          onchange={handleFiles}
          disabled={isOperationStarted}
      />
  {#if !formFiles}
    <Dialog.Description>
      For convenience, name files according to student number, e.g., "202011111.jpeg".
    </Dialog.Description>
  
  {:else if !isOperationStarted}
    <div class="space-y-1">
      <Dialog.Header class="flex flex-row justify-between">
        <span class="block">
          <Dialog.Title>
            Image preview
          </Dialog.Title>
          <Dialog.Description>
            Uploaded {formFileRecords.length} images
          </Dialog.Description>
        </span>
        <Button variant="secondary"
                  onclick={bulkUpload}>
          <IconSend class="size-4" />
        </Button>
      </Dialog.Header>
      <div class="border-b-2 border-t-2 border-border">
      <div class="flex flex-row items-center overflow-x-auto gap-x-1 -mx-6 px-6 pt-1 pb-3"
              style="scrollbar-gutter: stable; scrollbar-color: var(--secondary) transparent;">
        {#each formFileRecords as p}
          {@const supposedId = GET_NAME_ONLY(p.name)}
          {@const fileExtension = GET_EXTENSION_ONLY(p.name)}
          
          <div class="relative flex flex-col justify-end min-w-full sm:min-w-3/4 h-auto gap-y-0.5">
            <img src={p.url}
                  class="aspect-auto block"
                  alt={p.name} />
            <span class="absolute bottom-0 left-0 w-full flex flex-row px-2">
              <Dialog.Root>
                <Dialog.Trigger class="max-w-full flex flex-row gap-1.5 items-center truncate px-1.5 mb-1 bg-white/80 backdrop-blur-md 
                                    hover:underline cursor-pointer">
                  {#if IS_IN_STUDENTS(supposedId)}
                    <IconPerson class="size-4"/>
                  {:else}
                    <IconNotFound class="size-4" />
                  {/if}
                  <h4 class="text-left w-fit truncate">
                    {supposedId ? supposedId : "Add student no..."}
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
    </div>
  
  {:else}
    <div class="flex flex-col space-y-1">
      {#each formFileRecords as p}
        <span class="flex flex-row justify-start items-center gap-x-1.5">
          <span>
            {#if p.statusCode == 200}
              <IconCheck class="size-5"/>
            {:else if p.statusCode == 404}
              <IconNotFound class="size-5" />
            {:else if p.statusCode == 501}
              <IconExclamation class="size-5" />
            {:else if p.statusCode == -1}
              <Spinner class="text-primary size-4"/>
            {:else}
              <span class="font-mono -tracking-[0.08em]">{p.statusCode}</span>
            {/if}
          </span>
          <h4 class="truncate w-fit">{p.name}</h4>
        </span>
      {/each}
    </div>
  {/if}
</Dialog.Content>

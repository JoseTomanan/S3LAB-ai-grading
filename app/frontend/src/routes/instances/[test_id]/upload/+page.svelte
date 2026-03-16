<script lang="ts">
  let { data } = $props();

  import { API_BASE_URL } from '$lib/constants.ts';
  import { onMount } from 'svelte';

  const GET_NAME_ONLY = (s: string) => s.substring(0, s.lastIndexOf('.'));
  const GET_EXTENSION_ONLY = (s: string) => s.substring(s.lastIndexOf('.'));
  const IS_IN_STUDENTS = (x: string) => students.some(s => s.student_no == x)

  import IconCheck from "~icons/mdi/check";
  import IconExclamation from "~icons/mdi/exclamation-thick";
  import IconNotFound from "~icons/mdi/account-question-outline";
  import IconSend from "~icons/mdi/send";
  import IconPerson from "~icons/mdi/person";
  import IconRotateCW from "~icons/mdi/rotate-clockwise";
  import IconRotateCCW from "~icons/mdi/rotate-counter-clockwise";
  import IconFlipHorizontally from "~icons/mdi/flip-horizontal";
  import IconFlipVertically from "~icons/mdi/flip-vertical";

  import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import { Button } from '$lib/components/ui/button/index.ts';
  import { Input } from '$lib/components/ui/input/index.ts';
  import { Spinner } from '$lib/components/ui/spinner/index.ts';
  import BulkUploadRename from './BulkUploadRename.svelte';
  import type { FileRecord, Student } from '$lib/index.ts';

  import { rotateImage, flipImage } from '$lib/utils.ts';
  
  let isOperationStarted: boolean = $state(false);
  let formFiles: FileList | undefined = $state();
  let formFileRecords: FileRecord[] = $state([]);
  let students: Student[] = $state([]);


  onMount(async () => {
    try {
      const response = await fetch(`/api/sections/${data.test_instance!.section_id}`);
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
                `/api/student_answers/${data.test_id}/${activeStudentNo}/image_preprocess`,
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

  async function handleRotateCommand(formFileRecord: FileRecord, isCw: boolean) {
    if (!formFileRecord)
      return;
    const recordResponse = await rotateImage(formFileRecord, isCw);
    formFileRecords = formFileRecords.map(i => i.name == recordResponse.name ? recordResponse : i);
  }

  async function handleFlipCommand(formFileRecord: FileRecord, isFlipHorizontally: boolean) {
    if (!formFileRecord)
      return;
    const recordResponse = await flipImage(formFileRecord, isFlipHorizontally);
    formFileRecords = formFileRecords.map(i => i.name == recordResponse.name ? recordResponse : i);
  }
</script>


<div class="space-y-2">
  <h1 class="text-left font-semibold">Bulk upload</h1>
  <Input type="file"
          multiple
          accept="image/*"
          bind:files={formFiles}
          onchange={handleFiles}
          disabled={isOperationStarted}
      />
  
  {#if !formFiles}
    <h6 class="opacity-60 text-right">
      For convenience, name files according to student number, e.g., "202011111.jpeg".
    </h6>
  
  {:else if !isOperationStarted}
    <div class="space-y-1">
      <span class="flex flex-row justify-between items-center">
        <h5>
          Uploaded {formFileRecords.length} images
        </h5>
        <Button variant="secondary"
                  onclick={bulkUpload}>
          Send requests
          <IconSend class="size-4" />
        </Button>
      </span>
      <div class="border-b-2 border-t-2 border-border">
      <div class="flex flex-row items-center overflow-x-auto gap-x-1.5 -mx-6 px-6 pt-1 pb-3"
              style="scrollbar-gutter: stable; scrollbar-color: var(--chart-3) transparent;">
        {#each formFileRecords as p}
          {@const supposedId = GET_NAME_ONLY(p.name)}
          {@const fileExtension = GET_EXTENSION_ONLY(p.name)}
          
          <div class="relative flex flex-col justify-end min-w-3/4
                      sm:min-w-2/3 md:min-w-1/2 lg:min-w-1/3">
            <img src={p.url}
                  class="aspect-auto block"
                  alt={p.name} />
            <Dialog.Root>
              <Dialog.Trigger class="max-w-full flex flex-row gap-1.5 items-center bg-white/80 truncate px-1.5 backdrop-blur-md 
                                  hover:underline cursor-pointer
                                  absolute bottom-2 left-2">
                {#if IS_IN_STUDENTS(supposedId)}
                  <IconPerson class="size-4"/>
                {:else}
                  <IconNotFound class="size-4" />
                {/if}
                <h4 class="text-left w-fit truncate">
                  {supposedId ? supposedId : "Add student no..."}
                </h4>
              </Dialog.Trigger>
              <BulkUploadRename filename={supposedId}
                        onchange={(studentNo: string) => p.name = studentNo + fileExtension} />
            </Dialog.Root>
            <span class="absolute top-2 right-2 flex flex-row gap-x-1 opacity-80">
              <button onclick={() => handleRotateCommand(p, true)} class="button-outline" >
                <IconRotateCW />
              </button>
              <button onclick={() => handleRotateCommand(p, false)} class="button-outline" >
                <IconRotateCCW />
              </button>
              <button onclick={() => handleFlipCommand(p, true)} class="button-outline" >
                <IconFlipHorizontally />
              </button>
              <button onclick={() => handleFlipCommand(p, false)} class="button-outline" >
                <IconFlipVertically />
              </button>
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
</div>
<script lang="ts">
  let { data } = $props();

    import { onMount } from 'svelte';

  const GET_NAME_ONLY = (s: string) => s.substring(0, s.lastIndexOf('.'));
  const GET_EXTENSION_ONLY = (s: string) => s.substring(s.lastIndexOf('.'));
  const IS_IN_STUDENTS = (x: string) => students.some(s => s.student_no == x)

  // Supports: "201900000.jpeg" → "201900000"
  // Supports: "201900000-1.jpeg" → "201900000"
  const GET_STUDENT_NO = (s: string) => {
    const nameOnly = GET_NAME_ONLY(s);
    const dashIdx = nameOnly.lastIndexOf('-');
    if (dashIdx === -1) return nameOnly;
    const afterDash = nameOnly.substring(dashIdx + 1);
    return /^\d+$/.test(afterDash) ? nameOnly.substring(0, dashIdx) : nameOnly;
  };

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
  import { Label } from '$lib/components/ui/label/index.ts';
  import { Spinner } from '$lib/components/ui/spinner/index.ts';
  import { Separator } from '$lib/components/ui/separator/index.ts';
  import * as RadioGroup from '$lib/components/ui/radio-group/index.ts';
  import BulkUploadRename from './BulkUploadRename.svelte';
  import type { FileRecord, Student } from '$lib/index.ts';

  import { rotateImage, flipImage } from '$lib/utils.ts';
  
  let isOperationStarted: boolean = $state(false);
  let formFiles: FileList | undefined = $state();
  let formFileRecords: FileRecord[] = $state([]);
  let students: Student[] = $state([]);
  let segmentationStrategy: string = $state("corner_dots");
  let paperType: string = $state("ruled");
  let numBoxesPerStudent: Map<string, number | null> = $state(new Map());

  // TODO: finish API side, then wire API call
  const IS_FIRST_PAGE = (name: string): boolean => {
    const studentNo = GET_STUDENT_NO(name);
    const nameOnly = GET_NAME_ONLY(name);
    const dashIdx = nameOnly.lastIndexOf('-');
    if (dashIdx === -1) return true;
    const afterDash = nameOnly.substring(dashIdx + 1);
    return !(/^\d+$/.test(afterDash)) || afterDash === '1';
  };


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
      // Group files by student number
      const grouped = new Map<string, FileRecord[]>();
      for (const p of formFileRecords) {
        const studentNo = GET_STUDENT_NO(p.name);
        if (!grouped.has(studentNo)) grouped.set(studentNo, []);
        grouped.get(studentNo)!.push(p);
      }

      for (const [studentNo, records] of grouped) {
        if (!IS_IN_STUDENTS(studentNo)) {
          records.forEach(r => r.statusCode = 404);
          continue;
        }

        // Sort by page number for correct ordering
        records.sort((a, b) => {
          const aName = GET_NAME_ONLY(a.name);
          const bName = GET_NAME_ONLY(b.name);
          const aDash = aName.lastIndexOf('-');
          const bDash = bName.lastIndexOf('-');
          const aPage = aDash !== -1 ? parseInt(aName.substring(aDash + 1)) || 0 : 0;
          const bPage = bDash !== -1 ? parseInt(bName.substring(bDash + 1)) || 0 : 0;
          return aPage - bPage;
        });

        (async () => {
          const formData = new FormData();
          for (const r of records) {
            formData.append('files', r.file);
          }
          const response = await fetch(
                `/api/student_answers/${data.test_id}/${studentNo}/image_preprocess`,
                { method: "POST", body: formData, }
              );
          console.log(`${studentNo}: ${response.status}`);
          records.forEach(r => r.statusCode = response.status);
        })().catch(() => {
          records.forEach(r => r.statusCode = 500);
        });
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
  <div class="card grid grid-cols-2 gap-x-4">
    <div class="space-y-1.5">
      <Label>Segmentation strategy</Label>
      <RadioGroup.Root bind:value={segmentationStrategy}>
        <div class="flex items-center gap-x-1.5">
          <RadioGroup.Item value="corner_dots" />
          <Label>Corner dots</Label>
        </div>
        <div class="flex items-center gap-x-1.5">
          <RadioGroup.Item value="boxes" />
          <Label>Boxes</Label>
        </div>
      </RadioGroup.Root>
    </div>
    <div class="space-y-1.5">
      <Label>Paper type</Label>
      <RadioGroup.Root bind:value={paperType}>
        <div class="flex items-center gap-x-1.5">
          <RadioGroup.Item value="ruled" />
          <Label>Ruled</Label>
        </div>
        <div class="flex items-center gap-x-1.5">
          <RadioGroup.Item value="unruled" />
          <Label>Unruled</Label>
        </div>
      </RadioGroup.Root>
    </div>
  </div>
  <Separator/>
  
  {#if !formFiles}
    <h6 class="opacity-60 text-left">
      Uploads will appear here.
      <br>
      For ease, name by student no, e.g., 202011111.jpeg (single page) 202011111-1.jpeg (multi-page)
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
      <div class="flex flex-row items-center overflow-x-auto gap-x-1.5 -mx-6 px-6 pt-1 pb-3"
              style="scrollbar-gutter: stable; scrollbar-color: var(--chart-3) transparent;">
        {#each formFileRecords as p}
          {@const supposedId = GET_STUDENT_NO(p.name)}
          {@const nameOnly = GET_NAME_ONLY(p.name)}
          {@const fileExtension = GET_EXTENSION_ONLY(p.name)}
          {@const dashIdx = nameOnly.lastIndexOf('-')}
          {@const pageSuffix = dashIdx !== -1 && /^\d+$/.test(nameOnly.substring(dashIdx + 1)) ? nameOnly.substring(dashIdx) : ''}

          <div class="relative flex flex-col justify-end
                      min-w-full sm:min-w-2/3 md:min-w-1/2 lg:min-w-1/3">
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
                  {nameOnly ? nameOnly : "Add student no..."}
                </h4>
              </Dialog.Trigger>
              <BulkUploadRename filename={supposedId}
                        onchange={(studentNo: string) => p.name = studentNo + pageSuffix + fileExtension} />
            </Dialog.Root>
            {#if IS_FIRST_PAGE(p.name)}
              <span class="absolute bottom-2 right-2 bg-white/80 backdrop-blur-md">
                <input type="number"
                        class="w-16 px-1.5 py-0 leading-none bg-transparent outline-none [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
                        placeholder="# of box..."
                        value={numBoxesPerStudent.get(GET_STUDENT_NO(p.name)) ?? ''}
                        oninput={(e: Event) => {
                          const val = (e.target as HTMLInputElement).value;
                          numBoxesPerStudent = new Map(numBoxesPerStudent.set(
                            GET_STUDENT_NO(p.name),
                            val ? parseInt(val) : null
                          ));
                        }}
                    />
              </span>
            {/if}
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
      <Separator/>
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
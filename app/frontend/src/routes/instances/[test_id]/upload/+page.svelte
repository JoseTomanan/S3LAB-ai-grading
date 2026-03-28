<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  let { data } = $props();

  import { onMount } from 'svelte';

  const GET_NAME_ONLY = (s: string) => s.substring(0, s.lastIndexOf('.'));
  const GET_EXTENSION_ONLY = (s: string) => s.substring(s.lastIndexOf('.'));
  const IS_IN_STUDENTS = (x: string) => students.some(s => s.student_no == x)

  const GET_STUDENT_NO = (s: string) => {
    const nameOnly = GET_NAME_ONLY(s);
    const dashIdx = nameOnly.lastIndexOf('-');
    if (dashIdx === -1) return nameOnly;
    const afterDash = nameOnly.substring(dashIdx + 1);
    return /^\d+$/.test(afterDash) ? nameOnly.substring(0, dashIdx) : nameOnly;
  };

  import IconAccepted from "~icons/mdi/cloud-check-variant-outline";
  import IconExclamation from "~icons/mdi/exclamation-thick";
  import IconNotFound from "~icons/mdi/account-question-outline";
  import IconUnprocessable from "~icons/mdi/paper-off";

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
  import { Separator } from '$lib/components/ui/separator/index.ts';
  import BulkUploadRename from './BulkUploadRename.svelte';
  import type { FileRecord, Student } from '$lib/index.ts';

  import { handleRotateCommand, handleFlipCommand } from '$lib/utils/image_functions.ts';
	import toast from 'svelte-5-french-toast';
  
  let isOperationStarted: boolean = $state(false);
  let formFiles: FileList | undefined = $state();
  let formFileRecords: FileRecord[] = $state([]);
  let students: Student[] = $state([]);
  let numBoxesPerStudent: Map<string, number | null> = $state(new Map());

  const IS_FIRST_PAGE = (name: string): boolean => {
    const studentNo = GET_STUDENT_NO(name);
    const nameOnly = GET_NAME_ONLY(name);
    const dashIdx = nameOnly.lastIndexOf('-');
    if (dashIdx === -1)
      return true;
    const afterDash = nameOnly.substring(dashIdx + 1);
    return !(/^\d+$/.test(afterDash)) || afterDash === '1';
  };

  onMount(async () => {
    try {
      const response = await fetch(`${API_URL}/api/sections/${data.test_instance!.section_id}`);
      const results = await response.json();
      students = results;
    } catch (e) {
      console.log("Failed to fetch students for this section.");
    }
  });

  function handleFiles() {
    if (!formFiles || formFiles.length == 0)
      return;

    // FIXME: remove once validated that working as intended
    const validTypes = ['image/png', 'image/jpeg'];
    const isInvalid: boolean = Array.from(formFiles).some(
      (f) => !validTypes.includes(f.type)
      );
    if (isInvalid) {
      toast.error("Please upload only .png, .jpeg, or .jpg");
      return;
    }

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
                `${API_URL}/api/student_answers/${data.test_id}/${studentNo}/image_preprocess`,
                { method: "POST", body: formData, }
              );
          console.log(`${studentNo}: ${response.status}`);
          if (!response.ok) {
            const body = await response.json().catch(() => null);
            const detail: string = body?.detail ?? response.statusText;
            records.forEach(r => { r.statusCode = response.status; r.statusDetail = detail; });
          } else {
            records.forEach(r => r.statusCode = response.status);
          }
        })().catch(() => {
          records.forEach(r => r.statusCode = 500);
        });
      }
    } catch (e) {
      console.log("Bulk upload operation failed:\n"+e);
    }
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
  <Separator/>
  
  {#if !formFiles || formFiles.length == 0}
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
            <div class="absolute top-2 left-2 space-y-1">
              <Dialog.Root>
                <Dialog.Trigger class="max-w-full flex flex-row gap-1.5 items-center bg-white/80 truncate px-1.5 backdrop-blur-md
                                    hover:underline cursor-pointer">
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
              <span class="bg-white/80 backdrop-blur-md">
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
            </div>
            
            <span class="absolute top-2 right-2 flex flex-row gap-x-1 opacity-80">
              <button onclick={async () => formFileRecords = await handleRotateCommand(p, formFileRecords, true)}
                      class="button-outline">
                <IconRotateCW />
              </button>
              <button onclick={async () => formFileRecords = await handleRotateCommand(p, formFileRecords, false)}
                      class="button-outline">
                <IconRotateCCW />
              </button>
              <button onclick={async () => formFileRecords = await handleFlipCommand(p, formFileRecords, true)}
                      class="button-outline">
                <IconFlipHorizontally />
              </button>
              <button onclick={async () => formFileRecords = await handleFlipCommand(p, formFileRecords, false)}
                      class="button-outline">
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
        <span class="flex flex-row justify-start items-center gap-x-2
                      [&>span]:flex [&>span]:flex-row [&>span]:items-baseline [&>span]:size-5
                      [&>h4]:truncate [&>h4]:w-fit
                      [&>i]:text-base [&>i]:opacity-60 [&>i]:ml-2">
          <!-- <span> -->
            {#if p.statusCode == 200 || p.statusCode == 202}
              <span> <IconAccepted/> </span>
              <h4>{ p.name }</h4>
            {:else if p.statusCode == 404}
              <span> <IconNotFound/> </span>
              <h4>{ p.name }</h4>
              <i>Student no. not recognized</i>
            {:else if p.statusCode == 422}
              <span> <IconUnprocessable class="text-destructive"/> </span>
              <h4>{ p.name }</h4>
              <i>{p.statusDetail ?? "Unprocessable image"}</i>
            {:else if p.statusCode == 500}
              <span> <IconExclamation class="text-destructive"/> </span>
              <h4>{ p.name }</h4>
              <i>{p.statusDetail ?? "Unspecified server error"}</i>
            {:else if p.statusCode == 501}
              <span> <IconExclamation/> </span>
              <h4>{ p.name }</h4>
              <i>{p.statusDetail ?? ""}</i>
            
            {:else if p.statusCode == -1}
              <span> <Spinner class="text-primary"/> </span>
              <h4>{ p.name }</h4>
            {:else}
              <span class="font-mono -tracking-[0.08em]">{p.statusCode}</span>
              <h4>{ p.name }</h4>
              <i>{p.statusDetail ?? ""}</i>
            {/if}
          <!-- </span> -->
        </span>
      {/each}

    </div>
  {/if}
</div>
<script lang="ts">
  const { data } = $props();

  import { API_BASE_URL } from '$lib/constants.ts';
  import { dataUrlToFile } from '$lib/utils.ts';
  import OpenCamera from './OpenCamera.svelte';

  import IconCamera from "~icons/mdi/camera";
  import IconImagePreview from "~icons/mdi/image";
  import IconRotateCW from "~icons/mdi/rotate-clockwise";
  import IconRotateCCW from "~icons/mdi/rotate-counter-clockwise";
  import IconFlipHorizontally from "~icons/mdi/flip-horizontal";
  import IconFlipVertically from "~icons/mdi/flip-vertical";

  import * as Dialog from '$lib/components/ui/dialog/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Label } from '$lib/components/ui/label/index.ts';
	import { Spinner } from '$lib/components/ui/spinner/index.ts';
	import type { FileRecord } from '$lib/index.ts';

  import { rotateImage, flipImage } from '$lib/utils.ts';


  let isOperationOngoing: boolean = $state(false);

  let formFile: FileList | undefined = $state();
  let paramNumBoxes: number | null = $state(null);

  let formFileRecord: FileRecord | null = $state(null);

  let isAskingForValidation: boolean = $state(false);
  let supposedScans: {index: number, image_directory: string, item_number: string}[] = $state([]);

  function handleFile() {
    if (!formFile || formFile.length == 0)
      return;
    if (formFileRecord)
      URL.revokeObjectURL(formFileRecord.url);

    console.log("HANDLE FILE EXECUTING...");
    console.log(formFile[0].size);
    formFileRecord = {
      file: formFile[0],
      name: `${data.student_no}.jpeg`,
      url: URL.createObjectURL(formFile[0]),
      statusCode: -1,
    };
  }

  function getImageFromComponent(imageDataUrl: string) {
    const imageFile: File = dataUrlToFile(imageDataUrl, "CAPTURED_IMAGE.jpeg")
    formFile = undefined;
    formFileRecord = {
      file: imageFile,
      name: `${data.student_no}.jpeg`,
      url: URL.createObjectURL(imageFile),
      statusCode: -1,
    };
  }

  async function handleRotateCommand(isCw: boolean) {
    if (!formFileRecord)
      return;
    const recordResponse = await rotateImage(formFileRecord, isCw);
    formFileRecord = recordResponse;
  }

  async function handleFlipCommand(isFlipHorizontally: boolean) {
    if (!formFileRecord)
      return;
    const recordResponse = await flipImage(formFileRecord, isFlipHorizontally);
    formFileRecord = recordResponse;
  }

  async function sendImageForValidation() {
    if (!formFileRecord)
      return;

    console.log(formFileRecord.name);
    isOperationOngoing = true;

    const formData = new FormData();
    formData.append('file', formFileRecord.file);

    try {
      const response = await fetch(
            `${API_BASE_URL}/api/student_answers/${data.test_id}/${data.student_no}/label_save_boxes?num_boxes=${paramNumBoxes ?? 2}`,
            { method: "POST", body: formData, }
            );

      switch (response.status) {
        case 200:
          const result = await response.json();
          console.log(result);
          supposedScans = result.boxes ?? [];
          isAskingForValidation = true;
          break;
        default:
          alert(`${response.status} ${response.statusText}`);
      }
    } catch(e) {
      alert("Failed to send raw image for processing:\n"+e);
    } finally {
      isOperationOngoing = false;
    }
  }

  async function validateAndCommit(accept: boolean) {
    if (accept) {
      console.log(supposedScans);

      const response = await fetch(
            `${API_BASE_URL}/api/student_answers/${data.test_id}/${data.student_no}/commit_boxes`,
            {
              method: "POST",
              headers: {'Content-Type': 'application/json',},
              body: JSON.stringify({
                "boxes": supposedScans,
              }),
            }
          );
      switch (response.status) {
        case 200:
          alert("Review has been accepted.");
          window.location.reload();
          break;
        default:
          alert(`${response.status} ${response.statusText}`);
      }
      return;
    }

    // TODO: handle the otherwise case
  }
</script>


<div class="flex flex-col gap-y-2">
  <span class="flex flex-row justify-between items-baseline
            [&>h1]:font-semibold [&>h5]:opacity-60">
    <h1>Process image</h1>
    <h5>For {data.student_no}</h5>
  </span>

  {#if isAskingForValidation === false}
    <div class="flex flex-row gap-2 min-w-0">
      <Label for="sendImage"
              class="button-outline flex-1">
        <span class="shrink-0 whitespace-nowrap text-sm pl-1">
          {formFileRecord
            ? "Uploaded image:"
            : "Choose an image..." }
        </span>
        <span class="truncate min-w-0">
          {formFileRecord ? formFileRecord.name : ""}
        </span>
      </Label>
      <Input id="sendImage"
              type="file"
              accept="image/*"
              class="hidden"
              onchange={handleFile}
              bind:files={formFile}/>
      <Dialog.Root>
        <Dialog.Trigger class="button-outline w-1/5 h-auto flex justify-center items-center">
          <IconCamera class="size-6 opacity-80"/>
        </Dialog.Trigger>
        <OpenCamera onImageCapture={getImageFromComponent} />
      </Dialog.Root>
    </div>
    <Input id="numBoxes"
            type="number"
            placeholder="Number of boxes (default=2)..."
            bind:value={paramNumBoxes}
        />
    <Button variant="outline"
            onclick={sendImageForValidation}
            disabled={!formFileRecord || isOperationOngoing}>
      {isOperationOngoing
        ? "Sending..."
        : "Send for processing"}
      {#if isOperationOngoing}
        <Spinner />
      {/if}
    </Button>
    <div class="card flex flex-col items-center justify-center relative">
      {#if !formFileRecord}
        <IconImagePreview class="size-12 opacity-60"/>
        <h4 class="opacity-60">Uploaded image will be shown here.</h4>
      {:else}
        <img src={formFileRecord.url}
              alt="Uploaded file preview"
              class="md:max-w-3/4 lg:max-w-2/3"
              />
        <span class="flex flex-col gap-y-1 absolute top-2 right-2 size-fit">
          <button onclick={() => handleRotateCommand(true)} class="button-secondary" >
            <IconRotateCW />
          </button>
          <button onclick={() => handleRotateCommand(false)} class="button-secondary" >
            <IconRotateCCW />
          </button>
          <button onclick={() => handleFlipCommand(true)} class="button-secondary" >
            <IconFlipHorizontally />
          </button>
          <button onclick={() => handleFlipCommand(false)} class="button-secondary" >
            <IconFlipVertically />
          </button>
        </span>
      {/if}
    </div>

  {:else}
    <div class="flex flex-row w-full gap-x-1">
      <Button variant="outline"
              class="flex-1"
              onclick={() => validateAndCommit(true)}>
        Accept segmentation
      </Button>
      <button class="button-destructive text-sm"
              onclick={() => validateAndCommit(false)}>
        Discard
      </button>
    </div>
    
    {#each supposedScans as supposedScan}
      <div class="relative border border-border">
        <h4 class="absolute left-2 top-2 bg-white/80">
          {supposedScan.item_number}
        </h4>
        <img class="block"
              src={`${API_BASE_URL}${supposedScan.image_directory}`}
              alt={supposedScan.item_number}
              />
      </div>
    {/each}
  {/if}
</div>


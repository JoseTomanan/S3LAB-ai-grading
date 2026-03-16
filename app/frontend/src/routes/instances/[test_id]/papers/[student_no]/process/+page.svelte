<script lang="ts">
  const { data } = $props();

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
	import type { CommitBoxesResponseItem, FileRecord } from '$lib/index.ts';

  import { rotateImage, flipImage } from '$lib/utils.ts';
	import ReviewForCommit from './ReviewForCommit.svelte';


  let isCameraDialogOpen: boolean = $state(false);
  let isOperationOngoing: boolean = $state(false);
  let formFile: FileList | undefined = $state();
  let paramNumBoxes: number | null = $state(null);
  let formFileRecord: FileRecord | null = $state(null);
  let isAskingForValidation: boolean = $state(false);
  let supposedScans: (CommitBoxesResponseItem & { editing: boolean })[] = $state([]);

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
            `/api/student_answers/${data.test_id}/${data.student_no}/label_save_boxes?num_boxes=${paramNumBoxes ?? 2}`,
            { method: "POST", body: formData, }
            );

      switch (response.status) {
        case 200:
          const result = await response.json();
          console.log(result);
          supposedScans = (result.boxes ?? [])
                            .map((i: CommitBoxesResponseItem) => ({ ...i, editing: false }));
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
            `/api/student_answers/${data.test_id}/${data.student_no}/commit_boxes`,
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
          const responseBody = await response.json();
          alert(`${response.status}: ${responseBody.detail}`);
      }
      return;
    }

    // TODO: handle the otherwise case
  }

  // FIXME: test code, remove when no longer necessary
  // supposedScans =
  //       [
  //         { index: 1,
  //           image_directory: "/api/temp/3-Rizal_Seatwork-1_201911111_8a3ace9da3b04c48a3325e1507f620e5_0.jpg",
  //           item_number: "1", 
  //           editing: false },
  //         { index: 2,
  //           image_directory: "/api/temp/3-Rizal_Seatwork-1_201911111_9509b2b3fc9640419c5e3bf07c174d4d_2.jpg",
  //           item_number: "2", 
  //           editing: false },
  //         { index: 3,
  //           image_directory: "/api/temp/3-Rizal_Seatwork-1_201911111_2e6f8f201c5c4dec821d8d6b4537f5d3_1.jpg",
  //           item_number: "2b", 
  //           editing: false },
  //       ]
</script>


<div class="flex flex-col gap-y-2">
  <span class="flex flex-row justify-between items-baseline
            [&>h1]:font-semibold [&>h5]:opacity-60">
    <h1>Process image</h1>
    <h5>For {data.student_no}</h5>
  </span>
  <!-- {#if isAskingForValidation === false} -->
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
      <Dialog.Root bind:open={isCameraDialogOpen}>
        <Dialog.Trigger class="button-secondary w-1/5 h-auto flex justify-center items-center">
          <IconCamera class="size-6 opacity-80"/>
        </Dialog.Trigger>
        <OpenCamera {isCameraDialogOpen}
                    onImageCapture={getImageFromComponent} />
      </Dialog.Root>
    </div>
    <Input id="numBoxes"
            type="number"
            placeholder="Number of boxes (default=2)..."
            disabled={isAskingForValidation}
            bind:value={paramNumBoxes}
        />
  <!-- {/else} -->

  {#if isAskingForValidation === false}
  <!--FIXME: test code; remove when done-->
  <!-- {#if isAskingForValidation === true} -->
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
        <h6 class="opacity-60">Uploaded image will be shown here.</h6>
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
    <Dialog.Root>
      <Dialog.Trigger class="button-secondary text-sm">
        Open segmentation results
      </Dialog.Trigger>
      <ReviewForCommit supposedScans={supposedScans}
                        onAccept={() => validateAndCommit(true)}
                        onReject={() => validateAndCommit(false)} />
    </Dialog.Root>
    <h6>Note that segmentation results are ephemeral and will be disregarded when not accepted.</h6>
  {/if}
</div>


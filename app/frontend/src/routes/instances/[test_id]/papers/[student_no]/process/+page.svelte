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
  import * as RadioGroup from '$lib/components/ui/radio-group/index.ts';
	import type { CommitBoxesResponseItem, FileRecord } from '$lib/index.ts';

  import { rotateImage, flipImage } from '$lib/utils.ts';
	import ReviewForCommit from './ReviewForCommit.svelte';


  // TODO: Use `segmentationStrategy` and `supposedScans` as parameters in API calls


  let isCameraDialogOpen: boolean = $state(false);
  let isOperationOngoing: boolean = $state(false);
  // let formFile: FileList | undefined = $state();
  let formFiles: FileList | undefined = $state();
  let paramNumBoxes: number | null = $state(null);
  // let formFileRecord: FileRecord | null = $state(null);
  let formFileRecords: FileRecord[] = $state([]);
  let isAskingForValidation: boolean = $state(false);
  let segmentationStrategy: string = $state("corner_dots");
  let paperType: string = $state("ruled");
  let supposedScans: (CommitBoxesResponseItem & { editing: boolean })[] = $state([]);

  function handleFiles() {
    if (!formFiles || formFiles.length == 0)
      return;

    formFileRecords.forEach(r => URL.revokeObjectURL(r.url));
    formFileRecords = Array.from(formFiles).map((f, i) => ({
      file: f,
      name: `${data.student_no}_page${i + 1}.jpeg`,
      url: URL.createObjectURL(f),
      statusCode: -1,
    }));
  }

  function getImageFromComponent(imageDataUrl: string) {
    const imageFile: File = dataUrlToFile(imageDataUrl, "CAPTURED_IMAGE.jpeg")
    formFiles = undefined;
    formFileRecords = [...formFileRecords, {
      file: imageFile,
      name: `${data.student_no}_page${formFileRecords.length + 1}.jpeg`,
      url: URL.createObjectURL(imageFile),
      statusCode: -1,
    }];
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

  async function sendImageForValidation() {
    if (formFileRecords.length === 0)
      return;

    isOperationOngoing = true;

    const formData = new FormData();
    for (const r of formFileRecords) {
      formData.append('files', r.file);
    }

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

  let isCommitmentOngoing = $state(false);
  async function validateAndCommit(accept: boolean) {
    if (isCommitmentOngoing === true)
      return;

    isCommitmentOngoing = true;
    try {
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
    } catch {
      console.log("Catch kita kase nafall ka");
    } finally {
      isCommitmentOngoing = false;
    }
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
        class="button-outline flex-1 flex items-center gap-x-2 min-w-0">
        <span class="shrink-0 whitespace-nowrap text-sm pl-1">
          {formFileRecords.length > 0
            ? `Uploaded ${formFileRecords.length} image(s):`
            : "Choose image(s)..." }
        </span>
        {#if formFileRecords.length > 0}
          <span class="truncate min-w-0 flex-1">
            {formFileRecords.map(r => r.name).join(', ')}
          </span>
        {/if}
      </Label>
      <Input id="sendImage"
              type="file"
              accept="image/*"
              multiple
              class="hidden"
              onchange={handleFiles}
              bind:files={formFiles}/>
      <Dialog.Root bind:open={isCameraDialogOpen}>
        <Dialog.Trigger class="button-secondary w-1/5 h-auto flex justify-center items-center">
          <IconCamera class="size-6 opacity-80"/>
        </Dialog.Trigger>
        <OpenCamera {isCameraDialogOpen}
                    onImageCapture={getImageFromComponent} />
      </Dialog.Root>
    </div>
    <div class="flex flex-col sm:flex-row gap-x-4 gap-y-2">
      <Input id="numBoxes"
              class="flex-1"
              type="number"
              placeholder="Number of boxes (default: 2)..."
              disabled={isAskingForValidation}
              bind:value={paramNumBoxes}
          />
      <div class="card grid grid-cols-2 gap-x-4 flex-1">
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
    </div>
  <!-- {/else} -->

  {#if isAskingForValidation === false}
  <!--FIXME: test code; remove when done-->
  <!-- {#if isAskingForValidation === true} -->
    <Button variant="outline"
            onclick={sendImageForValidation}
            disabled={formFileRecords.length === 0 || isOperationOngoing}>
      {isOperationOngoing
        ? "Sending..."
        : "Send for processing"}
      {#if isOperationOngoing}
        <Spinner />
      {/if}
    </Button>

    <div class="card flex flex-col items-center justify-center">
      {#if formFileRecords.length === 0}
        <IconImagePreview class="size-12 opacity-60"/>
        <h6 class="opacity-60">
          Uploaded image(s) will be shown here.
        </h6>
      
      {:else}
        <div class="flex flex-row items-center overflow-x-auto gap-x-1.5 -mx-6 px-6 pt-1 pb-3"
              style="scrollbar-gutter: stable; scrollbar-color: var(--chart-3) transparent;">
          {#each formFileRecords as p}
            <div class="relative flex flex-col justify-end min-w-full sm:min-w-2/3 md:min-w-1/2 lg:min-w-1/3">
              <img src={p.url}
                    alt={p.name}
                    class="aspect-auto"
                    />
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
      {/if}
    </div>

  {:else}
    <Dialog.Root>
      <Dialog.Trigger class="button-secondary text-sm">
        Open segmentation results
      </Dialog.Trigger>
      <ReviewForCommit supposedScans={supposedScans}
                        {isCommitmentOngoing}
                        onAccept={() => validateAndCommit(true)}
                        onReject={() => validateAndCommit(false)}
                        />
    </Dialog.Root>
    <h6>Note that segmentation results are ephemeral and will be disregarded when not accepted.</h6>
  {/if}
</div>


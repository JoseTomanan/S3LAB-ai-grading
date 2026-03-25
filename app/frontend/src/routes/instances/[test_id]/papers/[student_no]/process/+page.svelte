<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  const { data } = $props();

  import { onDestroy } from 'svelte';
  import { goto, invalidateAll } from '$app/navigation';
  import { api, apiForm, ApiError } from '$lib/utils/api.ts';
  import toast from 'svelte-5-french-toast';
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
  import Switch from '$lib/LightSwitch.svelte';

	import type { CommitBoxesResponseItem, FileRecord } from '$lib/index.ts';
  import { createPoller } from '$lib/utils/poller.ts';
  import { rotateImage, flipImage } from '$lib/utils.ts';
	import ReviewForCommit from './ReviewForCommit.svelte';
	import { Separator } from '$lib/components/ui/separator/index.ts';

  let isCameraDialogOpen: boolean = $state(false);
  let isOperationOngoing: boolean = $state(false);
  
  let formFiles: FileList | undefined = $state();
  let paramNumBoxes: number | null = $state(null);
  let paramScannedAlready: boolean = $state(false);
  let formFileRecords: FileRecord[] = $state([]);
  
  let isAskingForValidation: boolean = $state(false);
  let isReviewDialogOpen: boolean = $state(false);
  let supposedScans: (CommitBoxesResponseItem & { editing: boolean })[] = $state([]);

  function handleFiles() {
    if (!formFiles || formFiles.length == 0)
      return;

    // Check all files: only allow .png, .jpeg, .jpg (case-insensitive)
    // FIXME: remove once validated that working as intended
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    const isInvalid: boolean = Array.from(formFiles).some(
      (f) => !validTypes.includes(f.type)
    );
    if (isInvalid) {
      toast.error("Please upload only .png, .jpeg, or .jpg");
      return;
    }

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

  // TODO: Use import
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

  // TODO: use paramScannedAlready in function. Do this after respective changes in backend are done
  async function sendImageForValidation() {
    if (formFileRecords.length === 0)
      return;

    isOperationOngoing = true;

    const formData = new FormData();
    for (const r of formFileRecords) {
      formData.append('files', r.file);
    }

    try {
      const result = await apiForm<{ boxes: CommitBoxesResponseItem[] }>(
            `${API_URL}/api/student_answers/${data.test_id}/${data.student_no}/label_save_boxes?num_boxes=${paramNumBoxes ?? 2}`,
            formData
            );
      supposedScans = (result.boxes ?? [])
                        .map((i: CommitBoxesResponseItem) => ({ ...i, editing: false }));
      isAskingForValidation = true;
      isReviewDialogOpen = true;
    } catch(e) {
      toast.error(e instanceof ApiError ? `${e.status} ${e.statusText}` : "Failed to send raw image for processing:\n" + e);
    } finally {
      isOperationOngoing = false;
    }
  }

  let isCommitmentOngoing = $state(false);

  const commitPoller = createPoller(async () => {
    const result = await api<{ statuses: { student_no: string, is_done_rendering: boolean }[] }>(
      `${API_URL}/api/student_answers/${data.test_id}/statuses`
    );
    const thisStudent = result.statuses.find(
      (s) => s.student_no === data.student_no
    );
    if (thisStudent?.is_done_rendering) {
      isCommitmentOngoing = false;
      await invalidateAll();
      goto(`/instances/${data.test_id}/papers/${data.student_no}`);
      return true;
    }
    return false;
  }, 5000);

  onDestroy(() => commitPoller.stop());

  async function validateAndCommit(accept: boolean) {
    if (isCommitmentOngoing === true)
      return;

    isCommitmentOngoing = true;
    try {
      if (accept) {
        await api(`${API_URL}/api/student_answers/${data.test_id}/${data.student_no}/commit_boxes`, {
          method: "POST",
          body: JSON.stringify({ boxes: supposedScans }),
        });

        isReviewDialogOpen = false;
        commitPoller.start();
        return;
      }

    // TODO: handle the otherwise case
    } catch (e) {
      toast.error(e instanceof ApiError ? `${e.status}: ${e.detail ?? e.statusText}` : "Failed to commit boxes:\n" + e);
    } finally {
      if (!commitPoller.active) isCommitmentOngoing = false;
    }
  }
</script>



<div class="flex flex-col gap-y-2 items-center">
  <span class="flex flex-row justify-between items-baseline w-full
                [&>h1]:font-semibold [&>h5]:opacity-60">
    <h1>Process image</h1>
    <h5>For {data.student_no}</h5>
  </span>

  <!-- <div> -->
    <div class="subcontainer flex flex-row gap-2 min-w-0">
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
              bind:files={formFiles}
          />
      
      <Dialog.Root bind:open={isCameraDialogOpen}>
        <Dialog.Trigger class="button-secondary w-1/5 h-auto flex justify-center items-center">
          <IconCamera class="size-6 opacity-80"/>
        </Dialog.Trigger>
        <OpenCamera {isCameraDialogOpen}
                    onImageCapture={getImageFromComponent} />
      </Dialog.Root>
    </div>
    
    <div class="subcontainer flex flex-col sm:flex-row gap-x-4 gap-y-2">
      <Input id="numBoxes" type="number"
              class="flex-2/3"
              placeholder="Number of boxes (default: 2)..."
              disabled={isAskingForValidation}
              bind:value={paramNumBoxes}
          />
      
      <span class="flex-1/3 flex flex-row gap-x-2 items-center">
        <Switch id="isAlreadyScanned"
                bind:checked={paramScannedAlready}/>
        <Label for="isAlreadyScanned">
          Page scanned already
        </Label>
      </span>
    </div>
  <!-- </div> -->

  <div class="subcontainer flex flex-col gap-y-2">
  {#if isAskingForValidation === false}
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

    <Separator/>
    <div class="subcontainer card flex flex-col items-center justify-center mx-auto">
      {#if formFileRecords.length === 0}
        <IconImagePreview class="size-12 opacity-60"/>
        <h6 class="opacity-60">
          Uploaded image(s) will be shown here.
        </h6>
      
      {:else}
        <div class="flex flex-row items-center overflow-x-auto gap-x-1.5 pt-1 pb-3
                    -mx-6 px-6 md:-mx-18 md:px-18 "
              style="scrollbar-gutter: stable; scrollbar-color: var(--chart-3) transparent;">
          {#each formFileRecords as p}
            <div class="relative flex flex-col justify-end
                        min-w-full sm:min-w-2/3 md:min-w-1/2 lg:min-w-1/3">
              <img src={p.url}
                    alt={p.name}
                    class="aspect-auto block"
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
    <Dialog.Root bind:open={isReviewDialogOpen}>
      <Dialog.Trigger class="button-secondary text-sm">
        Open segmentation results
      </Dialog.Trigger>
      <ReviewForCommit supposedScans={supposedScans}
                        {isCommitmentOngoing}
                        onAccept={() => validateAndCommit(true)}
                        onReject={() => validateAndCommit(false)}
                        />
    </Dialog.Root>
    {#if isCommitmentOngoing}
      <div class="flex flex-row items-center gap-x-2 text-sm text-muted-foreground">
        <Spinner class="size-4"/>
        Evaluating answers...
      </div>
    {:else}
      <h6>Note that segmentation results are ephemeral and will be disregarded when not accepted.</h6>
    {/if}
  {/if}
  </div>
</div>

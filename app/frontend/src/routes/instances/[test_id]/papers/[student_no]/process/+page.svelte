<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  const { data } = $props();

  import { onDestroy } from 'svelte';
  import { goto, invalidateAll } from '$app/navigation';
  import { api, apiForm, ApiError } from '$lib/utils/api.ts';
  import toast from 'svelte-5-french-toast';
  import { dataUrlToFile, isNotPngOrJpg } from '$lib/utils.ts';
  import OpenCamera from './OpenCamera.svelte';

  import IconUpload from "~icons/mdi/paperclip";
  import IconCamera from "~icons/mdi/camera";
  import IconImagePreview from "~icons/mdi/image";
  import IconRotateCW from "~icons/mdi/rotate-clockwise";
  import IconRotateCCW from "~icons/mdi/rotate-counter-clockwise";
  import IconFlipHorizontally from "~icons/mdi/flip-horizontal";
  import IconFlipVertically from "~icons/mdi/flip-vertical";

  import * as Tooltip from '$lib/components/ui/tooltip/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button, buttonVariants } from '$lib/components/CustomButton/index.js';
  import { cn } from '$lib/utils.ts';
  import { Label } from '$lib/components/ui/label/index.ts';
	import { Spinner } from '$lib/components/ui/spinner/index.ts';
  import Switch from '$lib/components/LightSwitch.svelte';
	import Card from '$lib/components/Card.svelte';

	import type { CommitBoxesResponseItem, FileRecord } from '$lib/index.ts';
  import { createPoller } from '$lib/utils/poller.ts';
  import { handleRotateCommand, handleFlipCommand } from '$lib/utils/image_functions.ts';
	import ReviewForCommit from './ReviewForCommit.svelte';
	import { Separator } from '$lib/components/ui/separator/index.ts';

  let isCameraDialogOpen: boolean = $state(false);
  let isOperationOngoing: boolean = $state(false);
  
  const defaultNumBoxes = $derived(data.test_items?.length ?? 2);

  let formFiles: FileList | undefined = $state();
  let paramNumBoxes: number | null = $state(null);
  let paramScannedAlready: boolean = $state(false);
  let formFileRecords: FileRecord[] = $state([]);
  
  let isAskingForValidation: boolean = $state(false);
  let isReviewDialogOpen: boolean = $state(false);
  let supposedScans: (CommitBoxesResponseItem & { editing: boolean; isDiscarded: boolean })[] = $state([]);

  function handleFiles() {
    if (!formFiles || formFiles.length == 0)
      return;

    if (isNotPngOrJpg(formFiles)) {
      toast.error("Please upload only .png, .jpeg, or .jpg");
      formFiles = undefined;
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


  async function sendImageForValidation() {
    if (formFileRecords.length === 0)
      return;

    isOperationOngoing = true;

    const formData = new FormData();
    for (const r of formFileRecords) {
      formData.append('files', r.file);
    }

    try {
      const url = `${API_URL}/api/student_answers/${data.test_id}/${data.student_no}/label_save_boxes?num_boxes=${paramNumBoxes ?? defaultNumBoxes}&is_scanned_already=${paramScannedAlready}`;
      const result = await apiForm<{ boxes: CommitBoxesResponseItem[] }>(url, formData);
      supposedScans = (result.boxes ?? [])
                        .map((i: CommitBoxesResponseItem) => ({ ...i, editing: false, isDiscarded: false }));
      isAskingForValidation = true;
      isReviewDialogOpen = true;
    } catch(e) {
      toast.error(e instanceof ApiError
        ? (e.detail ? e.detail : `${e.status} ${e.statusText}`)
        : "Failed to send raw image for processing:\n" + String(e));
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

  function resetPage() {
    commitPoller.stop();
    isCommitmentOngoing = false;
    isAskingForValidation = false;
    supposedScans = [];
    formFileRecords = [];
    formFiles = undefined;
  }

  async function validateAndCommit(accept: boolean) {
    if (isCommitmentOngoing === true)
      return;

    isCommitmentOngoing = true;

    const boxes = supposedScans.map(s => ({
      ...s,
      index: (!accept || s.isDiscarded) ? -1 : s.index,
    }));

    try {
      await api(`${API_URL}/api/student_answers/${data.test_id}/${data.student_no}/commit_boxes`, {
        method: "POST",
        body: JSON.stringify({ boxes: boxes }),
      });

      isReviewDialogOpen = false;
      commitPoller.start();
      return;
    } catch (e) {
      toast.error(e instanceof ApiError
        ? `${e.status}: ${e.detail ?? e.statusText}`
        : "Failed to commit boxes:\n" + e);
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
              class={cn(buttonVariants({ variant: 'outline' }), 'flex-1 flex justify-start items-center gap-x-1.5 min-w-0')}>
        <IconUpload/>
        <span class="shrink-0 whitespace-nowrap text-sm">
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
        <Dialog.Trigger class={cn(buttonVariants({ variant: 'secondary' }), 'w-1/5 flex justify-center items-center')}>
          <IconCamera class="size-6 opacity-80"/>
        </Dialog.Trigger>
        <OpenCamera {isCameraDialogOpen}
                    onImageCapture={getImageFromComponent} />
      </Dialog.Root>
    </div>
    
    <div class="subcontainer flex flex-row gap-x-4 gap-y-2">
      <Input id="numBoxes" type="number"
              class="flex-1"
              placeholder="# of boxes (default {defaultNumBoxes})..."
              disabled={isAskingForValidation}
              bind:value={paramNumBoxes}
          />
      
      <span class="w-fit flex flex-row gap-x-2 items-center">
        <Switch id="isAlreadyScanned"
                disabled={isAskingForValidation}
                bind:checked={paramScannedAlready}
              />
        <Tooltip.Root>
          <Tooltip.Trigger>
            <Label for="isAlreadyScanned">Pre-scanned</Label>
          </Tooltip.Trigger>
          <Tooltip.Content>
            Document has already been scanned
          </Tooltip.Content>
        </Tooltip.Root>
      </span>
    </div>
  <!-- </div> -->

  <div class="subcontainer flex flex-col gap-y-2">
  {#if isAskingForValidation === false}
    <Button variant="outline"
            onclick={sendImageForValidation}
            disabled={formFileRecords.length === 0 || isOperationOngoing}>
      {isOperationOngoing
        ? "Processing, do not quit..."
        : "Send for processing"}
      {#if isOperationOngoing}
        <Spinner />
      {/if}
    </Button>

    <Separator/>
    <Card class="subcontainer flex flex-col items-center justify-center mx-auto">
      {#if formFileRecords.length === 0}
        <IconImagePreview class="size-12 opacity-60"/>
        <h6 class="opacity-60">
          Uploaded image(s) will be shown here.
        </h6>
      
      {:else}
        <!-- TODO: Add function to disregard individual item (i.e. just evaluate the rest) -->
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
                <Button variant="outline" onclick={async () => formFileRecords = await handleRotateCommand(p, formFileRecords, true)}>
                  <IconRotateCW />
                </Button>
                <Button variant="outline" onclick={async () => formFileRecords = await handleRotateCommand(p, formFileRecords, false)}>
                  <IconRotateCCW />
                </Button>
                <Button variant="outline" onclick={async () => formFileRecords = await handleFlipCommand(p, formFileRecords, true)}>
                  <IconFlipHorizontally />
                </Button>
                <Button variant="outline" onclick={async () => formFileRecords = await handleFlipCommand(p, formFileRecords, false)}>
                  <IconFlipVertically />
                </Button>
              </span>
            </div>
          {/each}
        </div>
      {/if}
    </Card>

  {:else}
    <Dialog.Root bind:open={isReviewDialogOpen}>
      <Dialog.Trigger class={cn(buttonVariants({ variant: 'secondary' }), 'text-sm')}
                      disabled={isCommitmentOngoing}>
        Open segmentation results
      </Dialog.Trigger>
      <ReviewForCommit supposedScans={supposedScans}
                        {isCommitmentOngoing}
                        onAccept={() => validateAndCommit(true)}
                        onReject={() => validateAndCommit(false)}
                        />
    </Dialog.Root>
    <h6>
      <Spinner class={isCommitmentOngoing ? 'inline' : 'hidden'}/>
      {isCommitmentOngoing
        ? "Answers are being evaluated in the background. You may go back to the Papers screen now."
        : "Note that segmentation results are ephemeral, and switching out of this screen will discard them."}
    </h6>

    {#if isCommitmentOngoing}
      <Button variant="outline" onclick={resetPage}>
        Reset page
      </Button>
    {/if}

  {/if}
  </div>
</div>

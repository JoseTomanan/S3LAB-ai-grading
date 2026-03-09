<script lang="ts">
  const { data } = $props();

  import { API_BASE_URL } from '$lib/constants.ts';
  import { dataUrlToFile } from '$lib/utils.ts';
  import OpenCamera from './OpenCamera.svelte';

  import MdiCamera from "~icons/mdi/camera";

  import * as Dialog from '$lib/components/ui/dialog/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Label } from '$lib/components/ui/label/index.ts';
	import { Spinner } from '$lib/components/ui/spinner/index.ts';


  let isOperationOngoing: boolean = $state(false);

  let formFile: FileList | undefined = $state();
  let paramNumBoxes: number | null = $state(null);
  let uploadableFile: File | null = $state(null);
  $effect(() => {
    if (formFile)
      uploadableFile = formFile[0];
  });


  let isAskingForValidation: boolean = $state(false);

  let supposedScans: string[] = $state([]);


  function getImageFromComponent(imageDataUrl: string) {
    const imageFile: File = dataUrlToFile(imageDataUrl, "CAPTURED_IMAGE.jpeg")
    formFile = undefined;
    uploadableFile = imageFile;
  }

  async function sendImage() {
    if (!uploadableFile)
      return;

    console.log(uploadableFile.name);
    console.log(uploadableFile.size);
    isOperationOngoing = true;

    const formData = new FormData();
    formData.append('file', uploadableFile);

    try {
      const response = await fetch(
            `${API_BASE_URL}/api/student_answers/${data.test_id}/${data.student_no}/image_preprocess?num_boxes=${paramNumBoxes ?? 2}`,
            { method: "POST", body: formData, }
            );

      switch (response.status) {
        case 200:
          alert("Image has been processed and segmented into the appropriate items.");
          window.location.reload();
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

  async function sendImageForValidation() {
    if (!uploadableFile)
      return;

    console.log(uploadableFile.name);
    console.log(uploadableFile.size);
    isOperationOngoing = true;

    const formData = new FormData();
    formData.append('file', uploadableFile);

    // TODO: rest of function
  }

  async function validateAndCommit(accept: boolean) {
    // TODO: function
  }
</script>


<div class="flex flex-col gap-y-2">
  <span class="flex flex-row justify-between items-baseline [&>h5]:opacity-60">
    <h1>Process image</h1>
    <h5>Student no. {data.student_no}</h5>
  </span>
  {#if isAskingForValidation === false}
    <div class="flex flex-row gap-2 min-w-0">
      <Label for="sendImage"
              class="flex-1 border-2 border-outline pl-2 rounded
                  flex flex-row items-center min-w-0 font-medium shadow-xs">
        <span class="shrink-0 whitespace-nowrap text-sm">
          {uploadableFile
              ? "Uploaded image:"
              : "Choose an image..." }
        </span>
        <span class="truncate min-w-0">
          {uploadableFile ? uploadableFile.name : ""}
        </span>
      </Label>
      <Input id="sendImage"
              type="file"
              accept="image/*"
              class="hidden"
              bind:files={formFile}/>
      <Dialog.Root>
        <Dialog.Trigger class="button-secondary w-1/5 h-auto flex justify-center items-center">
          <MdiCamera class="size-6 opacity-80"/>
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
            disabled={!uploadableFile || isOperationOngoing}>
      {isOperationOngoing
        ? "Sending..."
        : "Send for processing"}
      {#if isOperationOngoing}
        <Spinner />
      {/if}
    </Button>

  {:else}
    <h3>Scanned document</h3>
    <img src=""
          alt="Scanned document"
          />
    <h3>Detected labels:</h3>
    <h5>haha</h5>
    <div class="w-full">
      <Button variant="outline"></Button>
      <button class="button-destructive">
        Discard
      </button>
    </div>
  {/if}
</div>


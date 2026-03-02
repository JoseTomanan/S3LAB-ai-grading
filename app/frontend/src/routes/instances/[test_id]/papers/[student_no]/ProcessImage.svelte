<script lang="ts">
  const { test_id, student_no }: { test_id: string, student_no: string } = $props();
  
  import { API_BASE_URL } from '$lib/constants.ts';
  import { dataUrlToFile } from '$lib/utils.ts';
  import OpenCamera from './OpenCamera.svelte';

  import MdiCamera from "~icons/mdi/camera";

  import * as Dialog from '$lib/components/ui/dialog/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Label } from '$lib/components/ui/label/index.ts';

  let isOperationOngoing: boolean = $state(false);

  let formFile: FileList | undefined = $state();
  let paramNumBoxes: number | null = $state(null);
  let uploadableFile: File | null = $derived(formFile ? formFile[0] : null);

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
            `${API_BASE_URL}/api/test_instances/${test_id}/${student_no}/image_preprocess?num_boxes=${paramNumBoxes ?? 2}`,
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
</script>


<Dialog.Content>
  <Dialog.Header>
    <Dialog.Title>Process raw image</Dialog.Title>
    <Dialog.Description>{test_id} &middot; {student_no}</Dialog.Description>
  </Dialog.Header>
  <div class="flex flex-row gap-1 min-w-0">
    <Label for="sendImage"
            class="grow button-outline flex flex-row items-center min-w-0 font-medium">
      <span class="shrink-0 whitespace-nowrap">
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
      <Dialog.Trigger class="button-secondary w-1/4 h-fit flex justify-center items-center">
        <MdiCamera class="size-6 my-0.5 opacity-80"/>
      </Dialog.Trigger>
      <OpenCamera onImageCapture={getImageFromComponent} />
    </Dialog.Root>
  </div>
  <Input id="numBoxes"
          type="number"
          placeholder="Number of boxes (default=2)..."
          bind:value={paramNumBoxes}/>
  <Button variant="outline"
          onclick={() => sendImage()}
          disabled={!uploadableFile}>
    Send for processing
  </Button>
  {#if isOperationOngoing}
    <p>Loading...</p>
  {/if}
</Dialog.Content>


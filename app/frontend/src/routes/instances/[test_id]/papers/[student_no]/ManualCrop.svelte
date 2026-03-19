<script lang="ts">
  import MdiUpload from '~icons/mdi/upload';

  import { Cropper, type CropperInstance } from "svelte-cropper";
  import { Input } from '$lib/components/ui/input/index.ts';
  import { Button } from '$lib/components/ui/button/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.js';

  let { test_id, student_no, item_id } = $props<{
    test_id: string,
    student_no: string,
    item_id: number
  }>();

  let isOperationOngoing: boolean = $state(false);

  let canvasCropper: CropperInstance | null = $state(null);
  let canvasFile: FileList | undefined = $state();
  let canvasImageUrl: string | null = $state(null);

  let responseImage: string = $state("");

  function handleFileUpload(e: Event) {
    const target = e.target as HTMLInputElement;
    const file = target.files?.[0];

    if (file) {
      if (canvasImageUrl)
        URL.revokeObjectURL(canvasImageUrl);
      canvasImageUrl = URL.createObjectURL(file);
    }
  }

  async function sendCropRequest() {
    isOperationOngoing = true;

    const canvasRectangle: {x: number, y: number, width: number, height: number} = canvasCropper!.getData();
    const returnablePoints: {x: number, y: number}[] = [
          { x: canvasRectangle.x, y: canvasRectangle.y },
          { x: canvasRectangle.x+canvasRectangle.width, y: canvasRectangle.y },
          { x: canvasRectangle.x+canvasRectangle.width, y: canvasRectangle.y+canvasRectangle.height },
          { x: canvasRectangle.x, y: canvasRectangle.y+canvasRectangle.height }
          ];

    const formData: FormData = new FormData();
    formData.append('file', canvasFile![0]);

    const formMetadata = {
          ul: { x: returnablePoints[0].x.toString(), y: returnablePoints[0].y.toString() },
          ur: { x: returnablePoints[1].x.toString(), y: returnablePoints[1].y.toString() },
          lr: { x: returnablePoints[2].x.toString(), y: returnablePoints[2].y.toString() },
          ll: { x: returnablePoints[3].x.toString(), y: returnablePoints[3].y.toString() },
          };
    formData.append('points', JSON.stringify(formMetadata));

    try {
      const response = await fetch(
            `/api/student_answers/${test_id}/${student_no}/${item_id}`,
            { method: "PATCH", body: formData, }
            );

      switch (response.status) {
        case 200:
          const result = await response.json();
          responseImage = result.image_directory;
          canvasImageUrl = null;
          alert("Addition successful.");
          window.location.reload();
          break;
        default:
          alert(`${response.status} ${response.statusText}`);
      }
    } catch(e) {
      alert("Failed to send points:\n"+e);
    } finally {
      isOperationOngoing = false;
    }
  }
</script>


<Dialog.Content class="max-w-3xl">
  <Dialog.Header>
    <Dialog.Title>
      Manually crop image
    </Dialog.Title>
    <Dialog.Description>
      {test_id} &middot; {student_no} &middot; ITEM_ID {item_id}
    </Dialog.Description>
  </Dialog.Header>
  <div class="flex flex-col space-y-2">
    <span class="flex flex-row space-x-1 w-full">
      <Input id="croppable"
            type="file" accept="image/*"
            bind:files={canvasFile}
            onchange={handleFileUpload}
            />
      <Button variant="secondary"
            disabled={!canvasFile || !canvasImageUrl || isOperationOngoing}
            onclick={() => sendCropRequest()}>
        Send
      </Button>
    </span>
    {#if isOperationOngoing}
      <p>Loading...</p>
    {/if}
    {#if canvasImageUrl}
      <div class="flex justify-center max-w-[95vh] md:max-h-[95vh] mx-auto">
        <Cropper bind:cropper={canvasCropper}
              src={canvasImageUrl}
              cropper_props={{viewMode: 2, dragMode: "crop", initialAspectRatio: 1}}
              />
      </div>
    {:else if responseImage != ""}
      <img class="border"
            src={responseImage} alt="Test item"/>
    {:else}
      <span class="py-4 border bg-muted text-muted-foreground flex flex-col items-center">
        <MdiUpload class="h-8 w-full"/>
        <p>Upload an image to start cropping.</p>
      </span>
    {/if}
  </div>
</Dialog.Content>

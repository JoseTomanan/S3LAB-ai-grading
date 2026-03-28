<script lang="ts">
  import { API_URL } from '$lib/constants.ts';
  import MdiUpload from '~icons/mdi/upload';

  import Cropper from 'cropperjs';
  import 'cropperjs/dist/cropper.css';

  import { apiForm, ApiError } from '$lib/utils/api.ts';
  import toast from 'svelte-5-french-toast';
  import { Input } from '$lib/components/ui/input/index.ts';
  import { Button } from '$lib/components/ui/button/index.ts';
  import * as Sheet from '$lib/components/ui/sheet/index.ts';
  import { Spinner } from '$lib/components/ui/spinner/index.ts';
	import Card from '$lib/components/Card.svelte';

  let { test_id, student_no, item_id, onCropSubmitted } = $props<{
    test_id: string,
    student_no: string,
    item_id: number,
    onCropSubmitted?: (imageDir: string) => void
  }>();

  let isOperationOngoing: boolean = $state(false);

  let canvasCropper: Cropper | null = $state(null);
  let canvasFile: FileList | undefined = $state();
  let canvasImageUrl: string | null = $state(null);

  let croppedPreviewUrl: string | null = $state(null);
  let responseImage: string = $state("");

  function handleCropperReady() {
    if (!canvasCropper) return;
    const canvasData = canvasCropper.getCanvasData();
    const boxSize = Math.min(canvasData.width, canvasData.height) * 0.1;
    canvasCropper.setCropBoxData({
      left: canvasData.left + canvasData.width - boxSize,
      top: canvasData.top + canvasData.height - boxSize,
      width: boxSize,
      height: boxSize
    });
  }

  function initCropper(node: HTMLImageElement, _src: string) {
    canvasCropper = new Cropper(node, {
      viewMode: 2,
      dragMode: 'crop',
      initialAspectRatio: 1,
      autoCropArea: 0.1,
      ready(event: Event) {
        const inst = (event.currentTarget as HTMLImageElement & { cropper: Cropper }).cropper;
        inst.zoomTo(0);
        inst.scaleY(1);
        inst.scaleX(1);
        inst.rotateTo(0);
        handleCropperReady();
      }
    });
    return {
      update(src: string) { canvasCropper?.replace(src); },
      destroy() {
        canvasCropper?.destroy(); canvasCropper = null;
        if (croppedPreviewUrl) { URL.revokeObjectURL(croppedPreviewUrl); croppedPreviewUrl = null; }
      }
    };
  }

  function handleFileUpload(e: Event) {
    const target = e.target as HTMLInputElement;
    const file = target.files?.[0];

    if (file) {
      if (croppedPreviewUrl) {
        URL.revokeObjectURL(croppedPreviewUrl);
        croppedPreviewUrl = null;
      }
      if (canvasImageUrl)
        URL.revokeObjectURL(canvasImageUrl);
      canvasImageUrl = URL.createObjectURL(file);
    }
  }

  let formData: FormData | null = $state(null);

  function cropImage() {
    if (!canvasCropper || !canvasFile?.[0]) return;

    formData = new FormData();
    formData.append('file', canvasFile[0]);

    const canvasRectangle: {x: number, y: number, width: number, height: number} = canvasCropper.getData();
    const returnablePoints: {x: number, y: number}[] = [
          { x: canvasRectangle.x, y: canvasRectangle.y },
          { x: canvasRectangle.x+canvasRectangle.width, y: canvasRectangle.y },
          { x: canvasRectangle.x+canvasRectangle.width, y: canvasRectangle.y+canvasRectangle.height },
          { x: canvasRectangle.x, y: canvasRectangle.y+canvasRectangle.height }
          ];

    const formMetadata = {
          ul: { x: returnablePoints[0].x.toString(), y: returnablePoints[0].y.toString() },
          ur: { x: returnablePoints[1].x.toString(), y: returnablePoints[1].y.toString() },
          lr: { x: returnablePoints[2].x.toString(), y: returnablePoints[2].y.toString() },
          ll: { x: returnablePoints[3].x.toString(), y: returnablePoints[3].y.toString() },
          };
    formData.append('points', JSON.stringify(formMetadata));

    const croppedCanvas = canvasCropper.getCroppedCanvas();
    croppedCanvas.toBlob((blob) => {
      if (blob) {
        if (croppedPreviewUrl) URL.revokeObjectURL(croppedPreviewUrl);
        croppedPreviewUrl = URL.createObjectURL(blob);
        canvasImageUrl = null;
      }
    });
  }

  async function sendCropRequest() {
    if (!formData) return;
    isOperationOngoing = true;
    try {
      const result = await apiForm<{ image_directory: string }>(
            `${API_URL}/api/student_answers/${test_id}/${student_no}/${item_id}`,
            formData,
            "PATCH"
            );
      responseImage = result.image_directory;
      if (croppedPreviewUrl) {
        URL.revokeObjectURL(croppedPreviewUrl);
        croppedPreviewUrl = null;
      }
      canvasImageUrl = null;
      onCropSubmitted?.(result.image_directory);
    } catch(e) {
      toast.error(e instanceof ApiError ? `${e.status} ${e.statusText}` : "Failed to send points:\n" + e);
    } finally {
      isOperationOngoing = false;
    }
  }
</script>


<Sheet.Content side="right"
                class="max-w-64">
  <Sheet.Header>
    <Sheet.Title>Manually crop image</Sheet.Title>
    <Sheet.Description>
      {test_id} &middot; {student_no} &middot; ITEM_ID {item_id}
    </Sheet.Description>
  </Sheet.Header>
  <div class="flex flex-col space-y-2 px-2 pb-4">
    <span class="flex flex-row space-x-1 w-full">
      <Input id="croppable"
            type="file" accept="image/*"
            bind:files={canvasFile}
            onchange={handleFileUpload}
            />
      {#if !croppedPreviewUrl}
        <Button variant="outline"
              disabled={!canvasFile || !canvasImageUrl}
              onclick={cropImage}>
          Crop image
        </Button>
      {:else}
        <Button variant="secondary"
              disabled={isOperationOngoing}
              onclick={() => sendCropRequest()}>
          {isOperationOngoing ? "Sending..." : "Send"}
        </Button>
      {/if}
    </span>

    {#if croppedPreviewUrl}
      <div class="relative flex justify-center max-w-[95vh] md:max-h-[95vh] mx-auto">
        <img class="border max-w-full"
              src={croppedPreviewUrl} alt="Cropped preview" />
        {#if isOperationOngoing}
          <div class="absolute top-1 left-1 bg-white/80">
            <Spinner class="size-16 text-chart-3" />
          </div>
        {/if}
      </div>
    
    {:else if canvasImageUrl}
      <div id="canvasArea"
            class="relative flex justify-center w-auto -mx-2">
        <img src={canvasImageUrl}
              use:initCropper={canvasImageUrl}
              alt="cropper_image"
              style="max-width: 100%; opacity: 0;"
            />
      </div>
      <h6>Scroll/pinch to zoom</h6>
    
    {:else if responseImage != ""}
      <img class="border"
            src={responseImage} alt="Test item"/>
    
    {:else}
      <Card class="py-4 border flex flex-col items-center">
        <MdiUpload class="h-8 w-full opacity-75"/>
        <p>Upload an image to start cropping.</p>
      </Card>
    {/if}
  </div>
</Sheet.Content>

<style>
  :global(.cropper-modal) {
    opacity: 0.20 !important;
  }
</style>

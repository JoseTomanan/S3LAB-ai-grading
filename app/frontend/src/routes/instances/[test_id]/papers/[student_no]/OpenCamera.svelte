<script lang="ts">
  let { onImageCapture } = $props();

  import EasyCamera from '@cloudparker/easy-camera-svelte';

  import MdiCamera from "~icons/mdi/camera";
  import MdiImageCheck from "~icons/mdi/image-check";

  import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import { Button } from '$lib/components/ui/button/index.ts';

  let cameraInstance: any = $state(null);
  let capturedImage: string = $state("");
  let isHasCaptured: boolean = $state(false);

  const handleImage = async () => {
          capturedImage = await cameraInstance.captureImage();
          console.log($state.snapshot(capturedImage));
          cameraInstance.close();
          isHasCaptured = true;
        };

  const returnImage = () => {
          // redownloadFile(dataUrlToFile(capturedImage, "camera input.jpeg"));
          onImageCapture(capturedImage);
        };
</script>


<Dialog.Content class="flex flex-col gap-4 items-center w-full h-full">
  {#if !isHasCaptured}
    <Dialog.Title>Capture image</Dialog.Title>
    <div class="w-full min-h-0 flex justify-center items-center overflow-hidden
            [&_video]:object-contain [&_video]:max-w-full [&_video]:max-h-full">
      <EasyCamera bind:this={cameraInstance}
                  useAudio={false}
                  width={720}
                  style="width: 100%; max-height: 100%; object-fit: contain;" />
    </div>
    <Button variant="outline"
              onclick={handleImage}
              class="px-4 py-2">
      <MdiCamera />
    </Button>
  {:else}
    <Dialog.Title>Preview image</Dialog.Title>
    <img src={capturedImage}
          alt="Captured preview"
          class="max-w-full max-h-full w-auto h-auto object-contain rounded-lg ring-4 ring-border shadow" />
    <Dialog.Close>
      <button class="button-outline"
                onclick={returnImage}>
        <MdiImageCheck />
      </button>
    </Dialog.Close>
  {/if}
</Dialog.Content>

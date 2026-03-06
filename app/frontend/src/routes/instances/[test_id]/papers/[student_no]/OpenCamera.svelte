<script lang="ts">
  let { onImageCapture } = $props();

  import MdiCamera from "~icons/mdi/camera";
  import MdiImageCheck from "~icons/mdi/image-check";
  
  import Camera from '$lib/components/MobileCamera/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import { Button } from '$lib/components/ui/button/index.ts';
	import { Skeleton } from "$lib/components/ui/skeleton/index.ts";
	// import { dataUrlToFile, redownloadFile } from "$lib/utils.ts";

  let cameraInstance: any = $state(null);
  let capturedImage: string = $state("");
  let isHasCaptured: boolean = $state(false);

  const handleImage = async () => {
          capturedImage = await cameraInstance.captureImage();
          console.log($state.snapshot(capturedImage));
          cameraInstance.close();
          isHasCaptured = true;
        };

  const retakeImage = () => {
          isHasCaptured = false;
          capturedImage = "";
          cameraInstance.open();
        };

  const returnImage = () => {
          // redownloadFile(dataUrlToFile(capturedImage, "camera input.jpeg"));
          onImageCapture(capturedImage);
        };

  let isCameraUnready: boolean = $state(false);
</script>


<Dialog.Content class="flex flex-col gap-4 items-center w-full h-full">
  <Dialog.Title>Capture image</Dialog.Title>
  {#if !isHasCaptured}
    <div class="w-full h-fit flex justify-center items-center overflow-hidden
            relative">
      <Camera bind:this={cameraInstance}
              useAudio={false}
              useFrontCamera={false}
              onOpen={() => isCameraUnready = false}
              onClose={() => isCameraUnready = true}
          />
      {#if isCameraUnready}
        <Skeleton class="absolute grayscale w-full h-full rounded-none"/>
      {/if}
    </div>
    <Button variant="outline"
              onclick={handleImage}
              class="px-4 py-2 w-full">
      <MdiCamera class="size-6"/>
    </Button>
  {:else}
    <img src={capturedImage}
          alt="Captured preview"
          class="border-0 ring-0 shadow" />
    <div class="flex flex-row w-full gap-1.5">
      <Dialog.Close class="flex-1 button-primary"
                  onclick={returnImage}>
        <MdiImageCheck class="size-6 w-full"/>
      </Dialog.Close>
      <Button variant="outline"
              onclick={retakeImage}>
        Retake
      </Button>
    </div>
  {/if}
</Dialog.Content>

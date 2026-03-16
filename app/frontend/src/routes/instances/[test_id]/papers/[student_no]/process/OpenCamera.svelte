<script lang="ts">
  let { isCameraDialogOpen, onImageCapture } = $props();

  import MdiCamera from "~icons/mdi/camera";
  import MdiCameraOff from "~icons/mdi/camera-off";
  import MdiImageCheck from "~icons/mdi/image-check";
  
	import DebugConsole from "$lib/components/DebugConsole.svelte";
  import Camera from '$lib/components/MobileCamera/index.ts';
  import * as Dialog from '$lib/components/ui/dialog/index.ts';
  import { Button } from '$lib/components/ui/button/index.ts';
	import { Skeleton } from "$lib/components/ui/skeleton/index.ts";
	import { Spinner } from "$lib/components/ui/spinner/index.ts";

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

  let isCameraUnready: boolean = $state(true);
  let isCameraDetected: boolean | null = $state(null);
  let cameraLoadTimeout: ReturnType<typeof setTimeout>;
  $effect(() => {
    if (!isCameraDialogOpen) {
      isCameraUnready = true;
      isCameraDetected = null;
      return;
    }

    cameraLoadTimeout = setTimeout(() => {
      console.error("Camera loading timed out.");
      if (isCameraUnready && isCameraDetected !== true)
        isCameraDetected = false;
    }, 5000);
    return () => clearTimeout(cameraLoadTimeout);
  });
</script>


<Dialog.Content class="flex flex-col gap-4 items-center w-full h-full">
  <Dialog.Title>Capture image</Dialog.Title>
  {#if !isHasCaptured}
    <div class="w-full h-fit flex justify-center items-center overflow-hidden
            relative {isCameraUnready && isCameraDetected !== false ? 'aspect-auto' : ''}">
      <Camera bind:this={cameraInstance}
              useAudio={false}
              useFrontCamera={false}
              onInit={(devices) => {
                  console.log("Executed onInit.")
                  isCameraDetected = devices.length>0;
                  }}
              onOpen={() => {
                  console.log("Executed onOpen.");
                  clearTimeout(cameraLoadTimeout);
                  isCameraDetected = true;
                  isCameraUnready = false;
                  }}
              onClose={() => isCameraUnready = true}
              />
      {#if isCameraDetected !== true}
        <div class="absolute inset-0 flex flex-row justify-center items-center opacity-75 space-x-1 font-semibold">
          {#if isCameraDetected === null}
              <Spinner />
              <h2>Camera loading...</h2>
          {:else}
              <MdiCameraOff/>
              <h2>Camera not detected</h2>
          {/if}
        </div>
      {:else if isCameraUnready}
        <Skeleton class="absolute inset-0 grayscale rounded-none"/>
      {/if}
    </div>
    <Button variant="outline"
              onclick={handleImage}
              disabled={isCameraUnready}
              class="px-4 py-2 w-full">
      <MdiCamera class="size-6"/>
    </Button>
  
  {:else}
    <img src={capturedImage}
          alt="Captured preview"
          class="border-0 ring-2 ring-accent rounded-xl" />
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
  
  <DebugConsole />
</Dialog.Content>
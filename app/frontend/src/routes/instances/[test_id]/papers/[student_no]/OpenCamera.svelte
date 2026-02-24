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
					onImageCapture(capturedImage);
				};
</script>

<Dialog.Content class="flex flex-col gap-4 items-center w-full h-full">
	{#if !isHasCaptured}
		<Dialog.Title>Capture image</Dialog.Title>
		<EasyCamera bind:this={cameraInstance}
								useAudio={false}
								width={400} />
		<Button variant="outline"
							onclick={handleImage}
							class="px-4 py-2">
			<MdiCamera />
		</Button>
	{:else}
		<Dialog.Title>Preview image</Dialog.Title>
		<img src={capturedImage}
					alt="Captured preview"
					class="w-[400px] rounded-lg ring-4 ring-border shadow" />
		<Dialog.Close>
			<button class="button-outline"
								onclick={returnImage}>
				<MdiImageCheck />
			</button>
		</Dialog.Close>
	{/if}
</Dialog.Content>

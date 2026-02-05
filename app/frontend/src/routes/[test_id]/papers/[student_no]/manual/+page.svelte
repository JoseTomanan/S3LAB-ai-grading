<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	const { data } = $props();

	import { Cropper, type CropperInstance } from "svelte-cropper";
	import { Input } from '$lib/components/ui/input/index.ts';
	import { Button } from '$lib/components/ui/button/index.ts';

	let isOperationOngoing: boolean = $state(false);

	let canvasCropper: CropperInstance | null = $state(null);
	let canvasFile: FileList | undefined = $state();
	let canvasImageUrl: string | null = $state(null);

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
				{ x: canvasRectangle.x, y: canvasRectangle.y },																								// Top-Left
				{ x: canvasRectangle.x+canvasRectangle.width, y: canvasRectangle.y },													// Top-Right
				{ x: canvasRectangle.x+canvasRectangle.width, y: canvasRectangle.y+canvasRectangle.height },	// Bottom-Right
				{ x: canvasRectangle.x, y: canvasRectangle.y+canvasRectangle.height }													// Bottom-Left
				];

		console.log("POINTS TO PASS: "+returnablePoints);

		const formData: FormData = new FormData();
		formData.append('image', canvasFile![0]);

		const formMetadata = { points: {
					ul: returnablePoints[0],
					ur: returnablePoints[1],
					lr: returnablePoints[2],
					ll: returnablePoints[3],
					}};
		formData.append('metadata', JSON.stringify(formMetadata));

		try {
			const response = await fetch(
						`${apiBaseUrl}/test_instances/${data.test_id}/${data.student_no}`,
						{ method: "PATCH", body: formData, }
						);

			const result = await response.json();
			canvasImageUrl = result.image_directory;
		} catch(e) {
			console.log("Failed to finish, check your network connection and try again.\n"+e);
		} finally {
			isOperationOngoing = false;
		}
	}
</script>


<div class="flex flex-col space-y-2">
	<h2 class="font-bold">Manually crop image</h2>
	<h6>{data.test_id} &middot; {data.student_no} &middot; ...</h6>
	<Input id="croppable"
				type="file" accept="image/*"
				bind:files={canvasFile}
				onchange={handleFileUpload}
				/>
	{#if canvasImageUrl}
		<Cropper bind:cropper={canvasCropper}
					src={canvasImageUrl}
					cropper_props={{viewMode: 2, dragMode: "crop", initialAspectRatio: 1}}
					/>
	{/if}
	{#if isOperationOngoing}
		<p>Loading...</p>
	{/if}
	<Button variant="secondary"
				disabled={!canvasFile || !canvasImageUrl || isOperationOngoing}
				onclick={() => sendCropRequest()}>
		Send crop request
	</Button>
</div>
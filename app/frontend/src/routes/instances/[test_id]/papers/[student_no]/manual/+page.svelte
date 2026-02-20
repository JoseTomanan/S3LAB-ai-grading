<script lang="ts">
	const { data } = $props();

	import { API_BASE_URL } from '$lib/constants.ts';
	import { page } from '$app/state';

	import MdiUpload from '~icons/mdi/upload';
	
	import { Cropper, type CropperInstance } from "svelte-cropper";
	import { Input } from '$lib/components/ui/input/index.ts';
	import { Button } from '$lib/components/ui/button/index.ts';
	
	const item_id = $derived(page.url.searchParams.get('item_id'));

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
						`${API_BASE_URL}/api/test_instances/${data.test_id}/${data.student_no}/${item_id}`,
						{ method: "PATCH", body: formData, }
						);

			switch (response.status) {
				case 200:
					const result = await response.json();
					responseImage = result.image_directory;
					canvasImageUrl = null;
					alert("Addition successful.");
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


<div class="flex flex-col space-y-2">
	<h2 class="font-bold">Manually crop image</h2>
	<h6>{data.test_id} &middot; {data.student_no} &middot; ITEM_ID {item_id}</h6>
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
	{#if canvasImageUrl}
		<Cropper bind:cropper={canvasCropper}
					src={canvasImageUrl}
					cropper_props={{viewMode: 2, dragMode: "crop", initialAspectRatio: 1}}
					/>
	{:else if responseImage != ""}
		<img class="border"
					src={API_BASE_URL+responseImage} alt="Test item"/>
	{:else}
		<span class="py-4 border bg-muted text-muted-foreground flex flex-col items-center">
			<MdiUpload class="h-8 w-full"/>
			<p>Upload an image to start cropping.</p>
		</span>
	{/if}
	{#if isOperationOngoing}
		<p>Loading...</p>
	{/if}
</div>
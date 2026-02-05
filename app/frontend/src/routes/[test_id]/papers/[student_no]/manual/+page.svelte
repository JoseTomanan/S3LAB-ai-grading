<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	const { data } = $props();

	import { Input } from '$lib/components/ui/input/index.ts';
	import { Button } from '$lib/components/ui/button/index.ts';

	let canvasFile: FileList | undefined = $state();
	let canvasImageUrl: string | null = $state(null);
	let canvasRectangle: {x: number, y: number, width: number, height: number};
	
	let returnablePoints: {x: number, y: number}[] = [];

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
		// TODO: function
	}
</script>


<div class="flex flex-col space-y-2">
	<h2 class="font-bold">Manually crop image</h2>
	<h6>{data.test_id} &middot; {data.student_no} &middot; ...</h6>
	<Input id="croppable"
				type="file" accept="image/*"
				bind:files={canvasFile}
				onchange={handleFileUpload}/>
	{#if canvasFile}
	<!-- TODO: all -->
	{/if}
	<Button variant="secondary" disabled={!canvasFile}
				onclick={() => sendCropRequest()}>
		Send crop request
	</Button>
</div>
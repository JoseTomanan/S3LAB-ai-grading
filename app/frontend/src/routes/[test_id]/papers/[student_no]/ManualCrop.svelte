<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
	
	const { test_id, student_no, studentItem }: { test_id: string, student_no: string, studentItem: (StudentAnswer & {label: string}) } = $props();

	import type { Student, StudentAnswer } from '$lib/index.ts';
	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Label } from '$lib/components/ui/label/index.js';

	let isOperationOngoing: boolean = $state(false);

	let formFile: FileList | undefined = $state();

	let manualCropPoints: {x: number, y: number}[] = $state([
				{x: 0, y: 0}, {x: 25, y: 0}, {x: 25, y: 25}, {x: 0, y: 25}, 
				]);

	$inspect(manualCropPoints);

	let canvasImageUrl: string | null = $state(null);
	let canvasActiveIndex: number | null = $state(null);
	let canvasImageElement: HTMLImageElement | null = $state(null);

	let canvasPolygonPath = $derived(manualCropPoints.map(p => `${p.x},${p.y}`).join(' '));

	function handleMouseMove(e: MouseEvent) {
		// TODO: function
	}

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


<Dialog.Content class="max-h-[95vh] overflow-y-auto">
	<Dialog.Title>Manually crop image</Dialog.Title>
	<Dialog.Description>{test_id} &middot; {student_no} &middot; Item {studentItem.label}</Dialog.Description>
	<Input id="sendCropRequest"
				type="file" accept="image/*"
				bind:files={formFile} onchange={handleFileUpload}
				/>
	{#if formFile}
		<div class="relative mt-1 inline-block overflow-hidden ring-2">
			<!-- TODO: create SVG canvas interface for picking 4 points -->
			<img src={canvasImageUrl}
						bind:this={canvasImageElement}
						alt="Croppable"
						draggable="false" />
			<svg viewBox="0 0 0 0">
				<polygon points={canvasPolygonPath} class="selection-area" />
				{#each manualCropPoints as point, i}
					<circle
								cx={point.x} cy={point.y}
								onmousedown={() => canvasActiveIndex = i}
								class="fill-black text-white border border-border cursor-move"
								role="button"
								tabindex="0"
					></circle>
				{/each}
			</svg>
		</div>
	{/if}
	<Button variant="default" onclick={() => sendCropRequest()} disabled={!formFile}>
		Send crop request
	</Button>
	{#if isOperationOngoing}
		<p>Loading...</p>
	{/if}
</Dialog.Content>


<style>
	/* TODO: add styles (take from Gemini's input) */
</style>
<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
	
	const { test_id, student_no, studentItem }: { test_id: string, student_no: string, studentItem: (StudentAnswer & {label: string}) } = $props();

	import type { Student, StudentAnswer } from '$lib/types/types.ts';
	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Label } from '$lib/components/ui/label/index.js';

	let isOperationOngoing: boolean = $state(false);

	let formFile: FileList | undefined = $state();

	let manualCropPoints: {x: number, y: number}[] = $state([
				{x: 0, y: 0}, {x: 25, y: 0}, {x: 25, y: 25}, {x: 0, y: 25}, 
				]);

	async function sendCropRequest() {
		// TODO: function
	}
</script>


<Dialog.Content>
	<Dialog.Title>Manually crop image</Dialog.Title>
	<Dialog.Description>{test_id} &middot; {student_no} &middot; Item {studentItem.label}</Dialog.Description>
	<Input id="sendCropRequest" type="file" accept="image/*" bind:files={formFile}/>
	{#if formFile}
		<div class="container">
			<!-- TODO: create SVG canvas interface for picking 4 points -->
		</div>
	{:else}
		<p>Please upload an image.</p>
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
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

	async function sendImage() {
		isOperationOngoing = true;

		if (!formFile || formFile.length == 0)
			return;

		const formData = new FormData();
    formData.append('image', formFile[0]);

		try {
			const response = await fetch(
				`${apiBaseUrl}/test_instances/cvprocess`,
				{ method: "POST", body: formData, }
			);

			if (response.ok) {
				alert("Image has been processed and replaced.");
				window.location.reload();
			} else
				alert(response.status + response.statusText);
		} catch(e) {
			alert("Failed to fetch, check your network connection and try again.\n"+e);
		} finally {
			isOperationOngoing = false;
		}
	}
</script>


<Dialog.Content>
	<Dialog.Title>Send image</Dialog.Title>
	<Dialog.Description>{test_id} &middot; {student_no} &middot; Item {studentItem.label}</Dialog.Description>
	<Input id="sendImage" type="file" accept="image/*" bind:files={formFile}/>
	<Button variant="default" onclick={() => sendImage()} disabled={!formFile}>
		Send for processing
	</Button>
	{#if isOperationOngoing}
		<p>Loading...</p>
	{/if}
</Dialog.Content>


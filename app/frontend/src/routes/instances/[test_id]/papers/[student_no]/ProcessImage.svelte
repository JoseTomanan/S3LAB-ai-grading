<script lang="ts">
	const { test_id, student_no }: { test_id: string, student_no: string } = $props();
	
	import { API_BASE_URL } from '$lib/constants.ts';

	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Label } from '$lib/components/ui/label/index.ts';

	let isOperationOngoing: boolean = $state(false);

	let formFile: FileList | undefined = $state();
	let paramNumBoxes: number | null = $state(null);

	async function sendImage() {
		isOperationOngoing = true;
		if (!paramNumBoxes || !formFile || formFile.length == 0)
			return;

		const formData = new FormData();
    formData.append('file', formFile[0]);

		try {
			// FIXME: Remove this line once backend reflects change in URI
			const response = await fetch(
						`${API_BASE_URL}/api/test_instances/${test_id}/${student_no}/image_preprocess?num_boxes=${paramNumBoxes}`,
						{ method: "POST", body: formData, }
						);

			switch (response.status) {
				case 200:
					alert("Image has been processed and segmented into the appropriate items.");
					window.location.reload();
					break;
				default:
					alert(`${response.status} ${response.statusText}`);
			}
		} catch(e) {
			alert("Failed to send raw image for processing:\n"+e);
		} finally {
			isOperationOngoing = false;
		}
	}
</script>


<Dialog.Content>
	<Dialog.Header>
		<Dialog.Title>Process raw image</Dialog.Title>
		<Dialog.Description>{test_id} &middot; {student_no}</Dialog.Description>
	</Dialog.Header>
	<Input id="sendImage"
					type="file"
					accept="image/*"
					bind:files={formFile}/>
	<Input id="numBoxes"
					type="number"
					placeholder="Number of boxes..."
					bind:value={paramNumBoxes}/>
	<Button variant="outline"
					onclick={() => sendImage()}
					disabled={!formFile}>
		Send for processing
	</Button>
	{#if isOperationOngoing}
		<p>Loading...</p>
	{/if}
</Dialog.Content>


<script lang="ts">
	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

	let {} = $props();

	import * as Dialog from "$lib/components/ui/dialog/index.js";

	import { Label } from '$lib/components/ui/label/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Button } from '$lib/components/ui/button/index.ts';

	// svelte-ignore non_reactive_update
	let newInstanceName: string = "";
	// svelte-ignore non_reactive_update
	let newInstanceSection: string = "";

	async function addNewTestInstance(name: string, section: string) {
		try {
			const response = await fetch(
				`${apiBaseUrl}/api/test_instances`,
				{
					method: "POST",
					headers: {'Content-Type': 'application/json',},
					body: JSON.stringify({
						"name": name,
						"section": section,
						"date": new Date().toISOString(),
					}),
				}
			);

			if (response.status == 200) {
				const data = response.json();
				alert("Addition successful.");
				window.location.reload();
			} else {
				alert(`Addition fail: ${response.status} ${response.statusText}`);
			}
		} catch (e) {
			alert("Failed to add new test instance. Check your network connection and try again.");
		}
	}
</script>


<Dialog.Content>
	<Dialog.Header>
		<Dialog.Title>Add new test instance</Dialog.Title>
	</Dialog.Header>
	<Label for="name">Test name</Label>
	<Input id="name" type="text"
					placeholder="Test name..."
					required
					bind:value={newInstanceName}/>
	<Label for="section">Section</Label>
	<Input id="section" type="text"
					placeholder="Section..."
					required
					bind:value={newInstanceSection}/>
	<Dialog.Footer>
		<Button variant="outline"
						onclick={() => addNewTestInstance(newInstanceName, newInstanceSection)}>
			Save changes
	</Button>
		<Dialog.Description>Note that the name and section cannot be changed after creation.</Dialog.Description>
	</Dialog.Footer>
</Dialog.Content>
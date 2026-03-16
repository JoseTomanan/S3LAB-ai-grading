<script lang="ts">
	import { API_BASE_URL } from '$lib/constants.ts';

	const { section_id } = $props();

	import * as Dialog from "$lib/components/ui/dialog/index.ts";
	import Button from '$lib/components/ui/button/button.svelte';
	import { Input } from '$lib/components/ui/input/index.ts';
	import { Label } from '$lib/components/ui/label/index.ts';

	let isLoading = $state(false);
	let formName: string = $state("");
	let formStudentNo: string = $state("");

	async function addNewStudent() {
		isLoading = true;
		try {
			const response = await fetch(
						`/api/students`,
						{
							method: "POST",
							headers: {'Content-Type': 'application/json',},
							body: JSON.stringify({
								"student_no": formStudentNo,
								"name": formName,
								"section_id": section_id,
							})
						}
						);

			switch (response.status) {
				case 201:
					const result = await response.json();
					alert("Success: "+result);
					window.location.reload();
					break;
				default:
					alert(`${response.status} ${response.statusText}`);
			}
		} catch (e) {
			alert("Failed to add new student:\nERROR: "+e);
		} finally {
			isLoading = false;
		}
	}
</script>


<Dialog.Content>
	<Dialog.Header>
		<Dialog.Title>Add new student</Dialog.Title>
	</Dialog.Header>
	<Label id="student_no">Student number</Label>
	<Input id="student_no"
					placeholder="Student number..."
					bind:value={formStudentNo}/>
	<Label id="name">Name</Label>
	<Input id="name"
					placeholder="Name..."
					bind:value={formName} />
	<Label id="section">Section</Label>
	<Input id="section"
					value={"SECTION_ID "+section_id}
					disabled/>
	<Dialog.Footer>
		<Button variant="outline"
						onclick={() => addNewStudent()}>
			Add new student
		</Button>
		{#if isLoading}
			<p>Loading...</p>
		{/if}
		<Dialog.Description>
			Note that the student number cannot be changed after creation.
		</Dialog.Description>
	</Dialog.Footer>
</Dialog.Content>
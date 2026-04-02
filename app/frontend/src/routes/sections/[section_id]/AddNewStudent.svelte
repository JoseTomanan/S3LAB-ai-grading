<script lang="ts">
	import { API_URL } from '$lib/constants.ts';

	let { section_id, isAddDialogOpen = $bindable() } = $props();

	import { invalidateAll } from '$app/navigation';
	import { api, ApiError } from '$lib/utils/api.ts';
	import toast from 'svelte-5-french-toast';
	import * as Dialog from "$lib/components/ui/dialog/index.ts";
	import Button from '$lib/components/CustomButton/button.svelte';
	import { Input } from '$lib/components/ui/input/index.ts';
	import { Label } from '$lib/components/ui/label/index.ts';

	let isLoading = $state(false);
	let formName: string = $state("");
	let formStudentNo: string = $state("");

	async function addNewStudent() {
    if (formStudentNo === "" || formName === "") {
      toast("Please do not leave empty fields");
      return;
    }

		isLoading = true;
		try {
			const result = await api(`${API_URL}/api/students`, {
				method: "POST",
				body: JSON.stringify({
					student_no: formStudentNo,
					name: formName,
					section_id: section_id,
				})
			});
			toast.success("Student added successfully!");
			await invalidateAll();
      isAddDialogOpen = false;
		} catch (e) {
			toast.error(e instanceof ApiError ? `${e.status} ${e.statusText}` : "Failed to add new student:\n" + e);
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

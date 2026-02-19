<script lang="ts">
	import { API_BASE_URL } from '$lib/constants.ts';

	const { section_id, student_no, name } = $props();

	import MdiDelete from "~icons/mdi/delete";

	import * as Dialog from "$lib/components/ui/dialog/index.ts";
	import Button from '$lib/components/ui/button/button.svelte';
	import { Input } from '$lib/components/ui/input/index.ts';
	import { Label } from '$lib/components/ui/label/index.ts';

	let isLoading = $state(false);
	let formName: string = $state(name);

	async function editStudent() {
		if (formName == name) {
			alert("No changes were made.");
			return;
		}
		// TODO: function
	}

	async function deleteStudent() {
		// TODO: function
	}
</script>


<Dialog.Content>
	<Dialog.Header>
		<Dialog.Title>Edit student</Dialog.Title>
	</Dialog.Header>
	<Label id="student_no">Student number</Label>
	<Input id="student_no"
					placeholder="Student number..."
					value={student_no}
					disabled/>
	<Label id="name">Name</Label>
	<Input id="name"
					placeholder="Name..."
					bind:value={formName} />
	<Dialog.Footer>
		{#if isLoading}
			<p>Loading...</p>
		{/if}
		<span class="flex flex-row gap-x-2">
			<Button variant="outline"
							class="w-4/5"
							onclick={() => editStudent()}>
				Save changes
			</Button>
			<Button variant="destructive"
								class="w-1/5"
								onclick={() => deleteStudent()}>
				<MdiDelete class="size-6"/>
			</Button>
		</span>
		<Dialog.Description>
			NOTE: Deletion cannot be undone!
		</Dialog.Description>
	</Dialog.Footer>
</Dialog.Content>

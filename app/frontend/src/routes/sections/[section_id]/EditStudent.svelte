<script lang="ts">
	import { API_BASE_URL } from '$lib/constants.ts';
	import { onMount } from 'svelte';

	const { section_id, student_no, name } = $props();

	import MdiDelete from "~icons/mdi/delete";

	import * as Dialog from "$lib/components/ui/dialog/index.ts";
	import * as Select from "$lib/components/ui/select/index.ts";
	import Button from '$lib/components/ui/button/button.svelte';
	import { Input } from '$lib/components/ui/input/index.ts';
	import { Label } from '$lib/components/ui/label/index.ts';
	
	import type { Section } from '$lib/index.ts';

	let isSectionsLoading = $state(false);
	let dropdownSections: Section[] = $state([]);

	let isLoading = $state(false);
	let formName: string = $state(name);
	let formSectionId: string = $state(section_id.toString())

	const triggerContent = $derived(
				dropdownSections.find((f) => f.section_id.toString() == formSectionId)?.section_name ?? "Section..."
				);

	onMount(async () => {
		isSectionsLoading = true;
		try {
			const response = await fetch(
						`${API_BASE_URL}/api/sections`,
						{
							method: "GET",
							headers: {'Content-Type': 'application/json',}
						}
						);

			switch (response.status) {
				case 200:
					const result = await response.json();
					dropdownSections = result;
					break;
				default:
					alert(`${response.status} ${response.statusText}`);
			}
		} catch (e) {
			alert("Failed to fetch sections.\n"+e);
		} finally {
			isSectionsLoading = false;
		}
	});

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
	<Label id="section">Section</Label>
	<Select.Root type="single"
								name="section"
								bind:value={formSectionId}>
		<Select.Trigger class="w-full text-base md:text-sm">
			{triggerContent}
		</Select.Trigger>
		<Select.Content>
			{#each dropdownSections as s (s.section_id)}
				<Select.Item value={s.section_id.toString()}
											label={s.section_name}>
					{s.section_name} ({s.section_id})
				</Select.Item>
			{/each}
		</Select.Content>
	</Select.Root>
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

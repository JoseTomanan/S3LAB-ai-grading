<script lang="ts">
	import { API_BASE_URL } from '$lib/constants.ts';
	import { onMount } from 'svelte';
	
	const { data } = $props();
	
	import MdiEdit from "~icons/mdi/edit";
	import type { Student } from '$lib/index.ts';
	import * as Dialog from '$lib/components/ui/dialog/index.ts';
	import AddNewStudent from './AddNewStudent.svelte';
	
	let isPageLoading = $state(true);
	let students: Student[] = $state([]);
	
	onMount(async () => {
		try {
			const response = await fetch(
						`${API_BASE_URL}/api/sections/${data.section_id}`,
						{
							method: "GET",
							headers: {'Content-Type': 'application/json',},
						}
						);

			switch (response.status) {
				case 200:
					const result = await response.json();
					students = result;
					break;
				default:
					alert(`${response.status} ${response.statusText}`);
			}
		} catch (e) {
			alert("Failed to fetch section details, students:\nERROR: "+e);
		} finally {
			isPageLoading = false;
		}
	}); 
</script>


<div class="px-4 py-8 flex flex-col gap-4">
	<span class="flex flex-row space-x-4 mb-4">
		<a href="/sections"
				class="font-bold button-outline">
			&middot;&middot;&middot;
		</a>
		<h1 class="align-left">
			From SECTION_ID {data.section_id}
		</h1>
	</span>
	<Dialog.Root>
		<Dialog.Trigger class="button-outline font-medium items-center">
			(+) Add new student
		</Dialog.Trigger>
		<AddNewStudent section_id={data.section_id}/>
	</Dialog.Root>
	{#if isPageLoading}
		<p>Loading...</p>
	{:else}
		{#each students as student}
			<div class="card">
				<span class="flex flex-row items-center justify-between">
					<h3>{student.name}</h3>
					<Dialog.Root>
						<Dialog.Trigger class="button-secondary">
							<MdiEdit class="size-4"/>
						</Dialog.Trigger>
						<!-- TODO: EditStudent functionality -->
						<!-- TODO: delete student call -->
					</Dialog.Root>
				</span>
				<h5>{student.student_no}</h5>
			</div>
		{/each}
	{/if}
</div>

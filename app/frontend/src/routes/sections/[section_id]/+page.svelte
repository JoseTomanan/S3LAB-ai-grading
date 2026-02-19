<script lang="ts">
	import { API_BASE_URL } from '$lib/constants.ts';
	import type { Student } from '$lib/index.ts';
	const { data } = $props();

	import { onMount } from 'svelte';

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
					console.log(result);
					students = result;
					break;
				default:
					alert(`${response.status} ${response.statusText}`);
			}
		} catch (e) {
			alert("Failed to fetch section details, students:\nERROR: "+e);
		} finally {

		}
	}); 
</script>


<div class="px-4 py-8 flex flex-col gap-4">
	<span class="flex flex-row justify-between">
		<h1>SECTION_ID {data.section_id}</h1>
		<a href="/sections"
				class="font-bold button-outline">
			&middot;&middot;&middot;
		</a>
	</span>
	<!-- TODO: add new student functionality -->
	<!-- TODO: edit student functionality -->
	<!-- TODO: delete student functionaity -->
	{#each students as student}
		<div class="card">
			<h3>{student.name}</h3>
			<h5>{student.student_no}</h5>
		</div>
	{/each}
</div>

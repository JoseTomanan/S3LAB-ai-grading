<script lang="ts">
	const { data } = $props();

	import { API_BASE_URL } from '$lib/constants.ts';
	import { onMount } from 'svelte';

	import MdiPaperOff from '~icons/mdi/paper-off';
	import MdiImagePlus from '~icons/mdi/image-plus';
	import MdiCrop from '~icons/mdi/crop';

	import type { StudentAnswer } from '$lib/index.ts';
	import * as Dialog from "$lib/components/ui/dialog/index.ts";
	import { Label } from '$lib/components/ui/label/index.js';
	import ProcessImage from './ProcessImage.svelte';

	let isPageLoading: boolean = $state(true);

	let studentItems: (StudentAnswer & {label: string})[] = $state([]);

	onMount(async () => {
		try {
			const response = await fetch(
						`${API_BASE_URL}/api/test_instances/${data.test_id}/${data.student_no}`,
						{
							method: "GET",
							headers: {'Content-Type': 'application/json',},
						}
					);
			
			switch (response.status) {
				case 200:
					const result = await response.json();
					studentItems = result.answers;
					break;
				case 204:
					console.log("--> 204");
					studentItems = [];
					break;
				default:
					alert(`${response.status} ${response.statusText}`);
			}

			// TODO: remove once backend code is fixed
			if (studentItems.length == 0)
				for (let i: number = 1; i < 5; i++)
					studentItems.push({
								answer_id: i,
								item_id: i,
								label: i.toString(),
								student_no: data.student_no,
								image_directory: "",
								ai_evaluation: "",
								is_done_rendering: false,
							});
		} catch(e) {
			alert("Failed to fetch, check your network connection and try again.\nERROR: "+e);
		} finally {
			isPageLoading = false;
		}
	});
</script>


<div class="flex flex-col gap-4 overflow-auto">
	<h2 class="font-semibold">For student {data.student_no}:</h2>
	{#if isPageLoading}
		<p>Loading...</p>
	{:else if studentItems.length == 0}
		<p>Nothing to see here. <br>If this is a mistake, check your network connection.</p>
	{:else}
		{#each studentItems as studentItem}
			<div class="card space-y-1">
				<Label for={studentItem.label} class="flex flex-row justify-between">
					{studentItem.label}
					<span class="flex flex-row space-x-1">
						<Dialog.Root>
							<Dialog.Trigger class="button-secondary border-none">
								<MdiImagePlus />
							</Dialog.Trigger>
							<ProcessImage test_id={data.test_id} student_no={data.student_no} {studentItem}/>
						</Dialog.Root>
						<a href={`/${data.test_id}/papers/${data.student_no}/manual?item_id=${studentItem.item_id}`} 
									class="button-secondary border-none">
							<MdiCrop/>
						</a>
					</span>
				</Label>
				<div class="flex justify-center items-center">
					{#if studentItem.image_directory == ""}
						<MdiPaperOff class="h-12 w-full py-2 bg-muted text-muted-foreground border" />
					{:else}
						<img class="size-fill max-h-[25vh]"
									src={`${API_BASE_URL}${studentItem.image_directory}`}
									alt={studentItem.label}/>
					{/if}
				</div>
			</div>
		{/each}
	{/if}
</div>

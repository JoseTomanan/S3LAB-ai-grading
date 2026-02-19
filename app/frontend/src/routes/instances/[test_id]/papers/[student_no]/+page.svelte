<script lang="ts">
	const { data } = $props();

	import { API_BASE_URL } from '$lib/constants.ts';
	import { onMount } from 'svelte';

	import MdiPaperOff from '~icons/mdi/paper-off';
	import MdiImagePlus from '~icons/mdi/image-plus';
	import MdiCrop from '~icons/mdi/crop';

	import type { TestItem, StudentAnswer } from '$lib/index.ts';
	import * as Dialog from "$lib/components/ui/dialog/index.ts";
	import { Label } from '$lib/components/ui/label/index.js';
	import ProcessImage from './ProcessImage.svelte';

	let isPageLoading: boolean = $state(true);

	let studentItems: (StudentAnswer & {label: string})[] = $state([]);
	let testItems: TestItem[] = $state([]);

	onMount(async () => {
		try {
			const responseOne = await fetch(
						`${API_BASE_URL}/api/test_instances/${data.test_id}/${data.student_no}`,
						{
							method: "GET",
							headers: {'Content-Type': 'application/json',},
						}
						);
			
			switch (responseOne.status) {
				case 200:
					const result = await responseOne.json();
					studentItems = result ? result : [];
					break;
				case 204:
					studentItems = [];
					break;
				default:
					alert(`CALL 1: ${responseOne.status} ${responseOne.statusText}`);
			}

			console.log($state.snapshot(studentItems));

			const responseTwo = await fetch(
						`${API_BASE_URL}/api/test_instances/${data.test_id}/items`,
						{
							method: "GET",
							headers: {'Content-Type': 'application/json',},
						}
						);

			switch (responseTwo.status) {
				case 200:
					const resultTwo = await responseTwo.json();
					testItems = resultTwo.items;
					console.log($state.snapshot(testItems));
					if (testItems && studentItems) {
						for (const testItem of testItems)
							if (!studentItems.find(si => si.item_id === testItem.item_id))
								studentItems.push({
											answer_id: 0,
											item_id: testItem.item_id,
											label: testItem.label,
											student_no: data.student_no,
											image_directory: "",
											ai_evaluation: "",
											is_done_rendering: false,
											});
					}
					break;
				default:
					alert(`CALL 2: ${responseTwo.status} ${responseTwo.statusText}`);
			}
		} catch(e) {
			alert("Failed to fetch student answers/test items:\nERROR: "+e);
		} finally {
			isPageLoading = false;
		}
	});
</script>


<div class="flex flex-col gap-4 overflow-auto p-0.5">
	<span class="flex flex-row justify-between items-center">
		<h1>{data.student_no}</h1>
		<Dialog.Root>
			<Dialog.Trigger class="button-outline">
				<MdiImagePlus class="size-6 mx-2"/>
			</Dialog.Trigger>
			<ProcessImage test_id={data.test_id} student_no={data.student_no} />
		</Dialog.Root>
	</span>
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
						<a href={`/instances/${data.test_id}/papers/${data.student_no}/manual?item_id=${studentItem.item_id}`} 
									class="button-secondary border-none">
							<MdiCrop/>
						</a>
					</span>
				</Label>
				<div class="flex justify-center items-center">
					{#if studentItem.image_directory == ""}
						<MdiPaperOff class="size-8 opacity-50" />
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

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
			
			// FIXME: there's an error here. 1 executes but 2 doesn't. Must be a backend thing.
			console.log("--> umabot dito 1");
			const result = await response.json();
			console.log("--> umabot dito 2");

			if (!(response.ok))
				alert(`${response.status} ${response.statusText}`);
			else
				studentItems = result.answers;
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
		{#each studentItems as item}
			<div class="card space-y-1">
				<Label for={item.label} class="flex flex-row justify-between">
					{item.label}
					<span class="flex flex-row space-x-1">
						<Dialog.Root>
							<Dialog.Trigger class="button-secondary border-none">
								<MdiImagePlus />
							</Dialog.Trigger>
							<ProcessImage test_id={data.test_id} student_no={data.student_no} studentItem={item}/>
						</Dialog.Root>
						<a href={`/${data.test_id}/papers/${data.student_no}/manual?item_id=${item.item_id}`} class="button-secondary border-none">
							<MdiCrop />
						</a>
					</span>
				</Label>
				<div class="flex justify-center items-center">
					{#if item.image_directory == ""}
						<MdiPaperOff class="h-12 w-full py-2 bg-muted text-muted-foreground border" />
					{:else}
						<img src={item.image_directory} class="size-fill max-h-[25vh]" alt={item.label}/>
					{/if}
				</div>
			</div>
		{/each}
	{/if}
</div>

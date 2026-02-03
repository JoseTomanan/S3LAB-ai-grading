<script lang="ts">
	const { data } = $props();

	import { onMount } from 'svelte';
	import MdiPaperOff from '~icons/mdi/paper-off';

	import type { StudentAnswer, TestItemHeader } from '$lib/types/types.ts';
	import * as Dialog from "$lib/components/ui/dialog/index.ts";
	import { Label } from '$lib/components/ui/label/index.js';
	import ProcessImage from '$lib/components/dialogs/ProcessImage.svelte';

	let isPageLoading: boolean = $state(true);

	let studentItems: (StudentAnswer & {label: string})[] = $state([]);

	onMount(async () => {
		// TODO: add fetch function
		try {
			const response = await fetch(``, {});
		} catch(e) {
			console.log("Failed to fetch, check your network connection and try again.\n"+e);
		} finally {
		}
	});

	// FIXME: remove (temporary!)
	setTimeout(() => {
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

		isPageLoading = false;
	}, 500);
</script>


<div class="flex flex-col gap-4">
	<h2 class="font-semibold">For student {data.student_no}:</h2>
	{#if isPageLoading}
		<p>Loading...</p>
	{:else}
		{#each studentItems as item}
			<Label for={item.label}>{item.label}</Label>
			<Dialog.Root>
				<Dialog.Trigger>
					<div class="flex justify-center items-center border border-border">
						{#if item.image_directory == ""}
							<MdiPaperOff class="size-6 my-2" />
						{:else}
							<img src="https://i.redd.it/wkijc5hpavt31.jpg" class="size-fill" alt={item.label}/>
						{/if}
					</div>
				</Dialog.Trigger>
				<ProcessImage student_no={data.student_no} studentItem={item}/>
			</Dialog.Root>
		{/each}
	{/if}
</div>